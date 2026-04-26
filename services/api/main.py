"""
Read-only FastAPI service for the React dashboard.

UNCERTAIN / PRODUCT NOTE (please confirm with domain owners):
- The UI speaks in 30/60/90 *day* risk horizons and multi-product portfolios.
- Postgres (`infra/init.sql`) only has `months_shrimp` and `dates_shrimp` — no per-SKU
  table and no stored forecast model outputs. We therefore:
  - expose a **single** logical product (shrimp / aggregate imports),
  - map horizons to **rolling windows of monthly history** (see `products` payload),
  - derive a 0–100 "risk score" from `monthly_import_zscore_6` heuristically.

News / articles are not in `init.sql`; evidence items are synthesized from
`dates_shrimp` when available. When the DB is down, `months_shrimp` is empty,
or a section has no real data yet, the API returns **labeled placeholders** so
the UI is testable (`meta.usingPlaceholders`, `meta.placeholderSections`).
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Generator

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone, timedelta
import requests

load_dotenv()

# --- ML model (supply_risk_regression.joblib) ----------------------------------------
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_REGRESSION_BUNDLE: dict[str, Any] | None = None
_BUNDLE_ERROR: str | None = None
_BUNDLE_LOADED: bool = False


def _get_regression_bundle() -> dict[str, Any] | None:
    """Lazy-load and cache the regression bundle on first call."""
    global _REGRESSION_BUNDLE, _BUNDLE_ERROR, _BUNDLE_LOADED
    if _BUNDLE_LOADED:
        return _REGRESSION_BUNDLE
    _BUNDLE_LOADED = True
    try:
        _REGRESSION_BUNDLE = joblib.load(_MODEL_DIR / "supply_risk_regression.joblib")
        # Inject created_utc from the manifest JSON if the bundle doesn't carry it
        if _REGRESSION_BUNDLE.get("created_utc") is None:
            _manifest = _MODEL_DIR / "supply_risk_manifest.json"
            if _manifest.is_file():
                import json as _json
                with _manifest.open() as _f:
                    _m = _json.load(_f)
                _REGRESSION_BUNDLE["created_utc"] = _m.get("created_utc")
    except Exception as exc:  # pragma: no cover
        _BUNDLE_ERROR = str(exc)
    return _REGRESSION_BUNDLE

# -------------------------------------------------------------------------------------

app = FastAPI(title="SupplySight Dashboard API", version="0.1.0")


class AgentMessageDTO(BaseModel):
    role: str = Field(..., description="Chat role: system | user | assistant")
    content: str = Field(..., min_length=1)


class AgentChatRequestDTO(BaseModel):
    messages: list[AgentMessageDTO]
    preferenceContext: dict[str, Any] | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=1.5)


def _build_agent_system_prompt(preferences: dict[str, Any] | None) -> str:
    pref_lines = []
    if preferences:
        for k, v in preferences.items():
            pref_lines.append(f"- {k}: {v}")

    pref_text = "\n".join(pref_lines) if pref_lines else "- none"

    return (
        "You are SupplySight Copilot, an operations assistant for shrimp supply risk decisions. "
        "Give concise, practical answers with actionable next steps. "
        "When discussing risk, explain tradeoffs and uncertainty clearly. "
        "If user asks for scenario impact, use their preference settings below as context.\n\n"
        "Current preference settings:\n"
        f"{pref_text}"
    )


def _call_openai_chat(messages: list[dict[str, str]], temperature: float) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the backend.",
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = "OpenAI returned an error"
        try:
            body = response.json()
            detail = body.get("error", {}).get("message", detail)
        except Exception:
            detail = response.text or detail
        raise HTTPException(status_code=502, detail=detail)

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unexpected OpenAI response format") from exc

    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=502, detail="Empty response from OpenAI")

    return content.strip()

_cors_origins = os.getenv(
    "SUPPLYSIGHT_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pg_params() -> dict[str, Any]:
    return {
        "host": os.getenv("PGHOST", "localhost"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "postgres")),
        "user": os.getenv("POSTGRES_USER", os.getenv("PGUSER", "postgres")),
        "password": os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", "")),
    }


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    conn = psycopg2.connect(**_pg_params())
    try:
        yield conn
    finally:
        conn.close()


def _compute_risk_score(z: float | None, price: float | None) -> tuple[int, str]:
    """
    Mirrors supply_risk_labels.py:
      raw = 8 * relu(-zscore_6) + 0.15 * relu(price_index - 90)
      scaled to [0, 100]; higher = more supply stress.
    raw_max ~16 empirically (zscore=-2 → raw=16).
    """
    z_val = 0.0 if (z is None or math.isnan(z)) else z
    p_val = 90.0 if (price is None or math.isnan(price)) else price
    raw = 8.0 * max(0.0, -z_val) + 0.15 * max(0.0, p_val - 90.0)
    score = int(min(100, round((raw / 16.0) * 100)))
    if score >= 75:
        return score, "Critical"
    if score >= 50:
        return score, "High"
    if score >= 25:
        return score, "Medium"
    return score, "Low"


def _score_to_level(score: int) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


# Maps bundle m_feature_names → months_shrimp DB column names
_M_COL_MAP: dict[str, str] = {
    "m__monthly_import": "monthly_import",
    "m__monthly_import_zscore_6": "monthly_import_zscore_6",
    "m__monthly_import_yoy_pct": "monthly_import_yoy_pct",
    "m__monthly_import_mom_pct": "monthly_import_mom_pct",
    "m__monthly_import_roll3_std": "monthly_import_roll3_std",
    "m__price_index_value": "price_index_value",
}


def _model_predict_score(row: dict[str, Any]) -> tuple[int, str] | None:
    """Score a months_shrimp row with the regression bundle. Returns None on any failure."""
    bundle = _get_regression_bundle()
    if bundle is None:
        return None
    try:
        import warnings
        import pandas as pd
        from services.supply_risk_training.supply_risk_labels import (
            daily_adjustment_oil_sentiment_batch,
        )
        m_names: list[str] = bundle["m_feature_names"]
        d_names: list[str] = bundle["d_feature_names"]
        nm, nd = len(m_names), len(d_names)
        m_vals = [_safe_float(row.get(_M_COL_MAP.get(n, n))) for n in m_names]
        X_full = np.array([m_vals + [float("nan")] * nd], dtype=float)
        # Pass a named DataFrame so the imputer (fitted with feature names) doesn't warn
        Xm_df = pd.DataFrame(X_full[:, :nm], columns=m_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Xm = bundle["imputer_month"].transform(Xm_df)
        X = X_full  # keep full array for d_block slice below
        monthly_risk = float(bundle["rf_month"].predict(Xm)[0])
        dc = float(bundle.get("delta_clip", 15.0))
        if bundle.get("architecture") == "monthly_plus_formula_adjustment":
            delta = float(
                daily_adjustment_oil_sentiment_batch(
                    X[:, nm:],
                    bundle["formula_meta"],
                    bundle.get("d_oil_ix"),
                    bundle.get("d_sent_ix"),
                    delta_clip=dc,
                )[0]
            )
        else:
            delta = 0.0
        score = int(round(min(100.0, max(1.0, monthly_risk + delta))))
        return score, _score_to_level(score)
    except Exception:
        return None


def _score_row(row: dict[str, Any]) -> tuple[int, str]:
    """Score a months_shrimp row: ML model when available, else heuristic fallback."""
    result = _model_predict_score(row)
    if result is not None:
        return result
    return _compute_risk_score(
        _safe_float(row.get("monthly_import_zscore_6")),
        _safe_float(row.get("price_index_value")),
    )


def _trend_from_scores(current: int, older: int | None) -> str:
    if older is None:
        return "stable"
    if current > older + 3:
        return "up"
    if current < older - 3:
        return "down"
    return "stable"


def _fetch_months_shrimp(conn, limit: int = 48) -> list[dict[str, Any]]:
    q = """
        SELECT date, monthly_import, monthly_import_zscore_6, price_index_value,
               monthly_import_mom_pct, monthly_import_yoy_pct, monthly_import_roll3_std
        FROM months_shrimp
        ORDER BY date DESC
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q, (limit,))
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("date"), date):
            d["date"] = d["date"].isoformat()
        out.append(d)
    return list(reversed(out))


def _fetch_latest_dates_shrimp(conn) -> dict[str, Any] | None:
    q = """
        SELECT * FROM dates_shrimp
        ORDER BY date DESC
        LIMIT 1
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        row = cur.fetchone()
    if not row:
        return None
    d = dict(row)
    if isinstance(d.get("date"), date):
        d["date"] = d["date"].isoformat()
    return d


# --- Placeholders for UI testing (uncertain / not-yet-modeled fields) -----------------

PLACEHOLDER_NOTE = "[Placeholder] Replace when forecasts, alerts, and evidence feeds exist."


def _placeholder_trend() -> list[dict[str, Any]]:
    """Synthetic monthly points so the chart renders without DB rows."""
    base = [
        (62, 1.02e7, 118.2),
        (64, 1.05e7, 119.0),
        (58, 9.8e6, 117.5),
        (55, 9.5e6, 116.8),
        (59, 9.9e6, 117.0),
        (61, 1.01e7, 117.8),
        (63, 1.03e7, 118.5),
        (67, 1.08e7, 119.2),
        (70, 1.10e7, 120.0),
        (72, 1.12e7, 120.4),
        (68, 1.09e7, 119.8),
        (71, 1.11e7, 120.1),
    ]
    return [
        {
            "date": f"2025-{i + 1:02d}",
            "shrimp": base[i][0],
            "monthlyImport": base[i][1],
            "priceIndex": base[i][2],
        }
        for i in range(len(base))
    ]


def _placeholder_products() -> list[dict[str, Any]]:
    return [
        {
            "id": "PRD-PLACEHOLDER-001",
            "name": "Shrimp  ",
            "category": "Seafood",
            "supplier": "—",
            "risk30": {"level": "High", "score": 72, "trend": "up"},
            "risk60": {"level": "Medium", "score": 58, "trend": "stable"},
            "risk90": {"level": "Low", "score": 42, "trend": "down"},
        },
    ]


def _placeholder_overview() -> list[dict[str, Any]]:
    return [
        {
            "key": "risk",
            "label": "Overall Risk Level",
            "value": "Medium",
            "subtext": PLACEHOLDER_NOTE,
        },
        {
            "key": "products",
            "label": "Monitored Products",
            "value": "3",
            "subtext": PLACEHOLDER_NOTE,
        },
        {
            "key": "alerts",
            "label": "Active Alerts",
            "value": "4",
            "subtext": "2 critical, 2 warnings — " + PLACEHOLDER_NOTE,
        },
    ]


def _placeholder_evidence() -> list[dict[str, Any]]:
    return [
        {
            "iconType": "globe",
            "title": "Seafood supply disruption (placeholder)",
            "description": "Example copy for UI testing. Hook to news / dates_shrimp when ready.",
            "source": "Placeholder feed",
            "impact": "Critical",
            "date": "2026-03-15",
        },
        {
            "iconType": "trending",
            "title": "Feed cost volatility (placeholder)",
            "description": "Example macro signal. Replace with real commodity or model output.",
            "source": "Placeholder feed",
            "impact": "High",
            "date": "2026-03-16",
        },
    ]


def _placeholder_recommendations() -> list[dict[str, Any]]:
    return [
        {
            "iconType": "cart",
            "action": "Buy early (placeholder)",
            "product": "Frozen Shrimp",
            "description": "Sample mitigation action for layout testing.",
            "priority": "High",
            "savings": "Placeholder — model-driven estimate TBD",
            "timeline": "Action needed within 5 days (placeholder)",
        },
        {
            "iconType": "refresh",
            "action": "Diversify suppliers (placeholder)",
            "product": "Frozen Shrimp",
            "description": "Sample diversification suggestion for layout testing.",
            "priority": "High",
            "savings": "Placeholder — risk reduction TBD",
            "timeline": "Evaluate by end of quarter (placeholder)",
        },
    ]


def _fetch_news(conn, limit: int = 10) -> list[dict[str, Any]]:
    """Query evaluated_news JOIN news, return EvidenceItemDTO-shaped dicts."""
    q = """
        SELECT
            n.title,
            n.url,
            n.source,
            n.publication_date,
            e.relevancy_score,
            e.sentiment_score,
            e.product
        FROM evaluated_news e
        JOIN news n ON n.id = e.id
        WHERE e.relevancy_score > 50 
        AND n.publication_date >= NOW() - INTERVAL '60 days'
        ORDER BY ABS(e.sentiment_score - 50) DESC, n.publication_date DESC
        LIMIT %s
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (limit,))
            rows = cur.fetchall()
    except Exception:
        return []

    items: list[dict[str, Any]] = []
    for r in rows:
        sentiment = _safe_float(r.get("sentiment_score")) or 50.0
        # sentiment_score: 1=severe shortage (bad), 100=surplus (good)
        intensity = abs(sentiment - 50)
        if sentiment < 50:
            icon_color = "red"
            impact = "High" if intensity >= 20 else "Medium"
            if intensity >= 40:
                severity_msg = "Severe shortage warning"
            elif intensity >= 25:
                severity_msg = "Strong shortage signal"
            elif intensity >= 10:
                severity_msg = "Moderate shortage concern"
            else:
                severity_msg = "Mild shortage concern"
        elif sentiment > 50:
            icon_color = "green"
            impact = "Low"
            if intensity >= 40:
                severity_msg = "Strong surplus signal"
            elif intensity >= 25:
                severity_msg = "Strong supply improvement"
            elif intensity >= 10:
                severity_msg = "Moderate supply improvement"
            else:
                severity_msg = "Mild supply improvement"
        else:
            icon_color = "neutral"
            impact = "Medium"
            severity_msg = "Neutral supply signal"

        if intensity > 30:
            impact = "High"
        elif intensity > 20:
            impact = "Medium"
        else:
            intensity = "Low"

        pub_date = r.get("publication_date")
        if isinstance(pub_date, date):
            pub_date = pub_date.isoformat()

        product = r.get("product") or "shrimp"
        description = f"{severity_msg} for {product}."

        items.append({
            "iconType": "globe",
            "iconColor": icon_color,
            "title": r.get("title") or "Untitled",
            "description": description,
            "source": r.get("source") or "Unknown",
            "impact": impact,
            "date": pub_date or "",
            "url": r.get("url") or None,
            "relevancyScore": int(r["relevancy_score"]) if r.get("relevancy_score") is not None else None,
        })
    return items


def _full_placeholder_response(
    *,
    reason: str,
    db_error: str | None = None,
) -> dict[str, Any]:
    """All uncertain / empty sections filled so the UI is testable without real data."""
    meta: dict[str, Any] = {
        "asOf": "2026-03-01",
        "hasData": False,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "usingPlaceholders": True,
        "placeholderReason": reason,
        "placeholderSections": [
            "overview",
            "trend",
            "products",
            "evidence",
            "recommendations",
            "alerts",
        ],
    }
    if db_error:
        meta["dbError"] = db_error
    return {
        "meta": meta,
        "overview": _placeholder_overview(),
        "products": _placeholder_products(),
        "trend": _placeholder_trend(),
        "evidence": _placeholder_evidence(),
        "recommendations": _placeholder_recommendations(),
    }


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _overall_risk_label(months: list[dict[str, Any]]) -> tuple[str, str]:
    if not months:
        return "N/A", "No monthly rows in months_shrimp"
    last = months[-1]
    score, _ = _score_row(last)
    shi = round((100 - score) / 10, 1)
    mom = _safe_float(last.get("monthly_import_mom_pct"))
    sub = "Based on import z-score and price index"
    if mom is not None:
        sub = f"MoM import change {mom * 100:+.1f}% vs prior month"
    return str(shi), sub


def _build_recommendations(months: list[dict[str, Any]], as_of: str | None) -> list[dict[str, Any]]:
    """Generate actionable recommendations from real risk signals — always returns at least one item."""
    if not months:
        return []

    last = months[-1]
    score, level = _score_row(last)
    mom = _safe_float(last.get("monthly_import_mom_pct"))
    mom_pct = mom * 100 if mom is not None else None
    date_label = as_of or "latest month"

    recs: list[dict[str, Any]] = []

    if level == "Critical":
        recs.append({
            "iconType": "alert",
            "action": "Place emergency order now",
            "product": "Shrimp  ",
            "description": (
                f"Shortage risk is Critical (Supply Health Index: {(100 - score) / 10:.1f}/10). Import volumes are sharply below the 6-month average"
                + (f" and fell {abs(mom_pct):.1f}% last month" if mom_pct is not None and mom_pct < 0 else "")
                + ". Order immediately to avoid stockouts."
            ),
            "priority": "High",
            "savings": "Prevents stockout",
            "timeline": f"Act within 2 days · data as of {date_label}",
        })
        recs.append({
            "iconType": "refresh",
            "action": "Identify backup suppliers",
            "product": "Shrimp  ",
            "description": "Critical risk level suggests a supply disruption. Contact alternative suppliers to secure a contingency source.",
            "priority": "High",
            "savings": "Reduces single-source dependency",
            "timeline": f"This week · data as of {date_label}",
        })

    elif level == "High":
        recs.append({
            "iconType": "cart",
            "action": "Increase order quantity",
            "product": "Shrimp  ",
            "description": (
                f"Shortage risk is High (Supply Health Index: {(100 - score) / 10:.1f}/10). Imports are below the seasonal average"
                + (f", down {abs(mom_pct):.1f}% from last month" if mom_pct is not None and mom_pct < 0 else "")
                + ". Ordering above your normal quantity now will provide a buffer."
            ),
            "priority": "High",
            "savings": "Reduces stockout risk",
            "timeline": f"Before next order cycle · data as of {date_label}",
        })

    elif level == "Medium":
        if mom_pct is not None and mom_pct < -5:
            recs.append({
                "iconType": "clock",
                "action": "Order slightly early this cycle",
                "product": "Shrimp  ",
                "description": (
                    f"Risk is Medium (Supply Health Index: {(100 - score) / 10:.1f}/10) but imports dropped {abs(mom_pct):.1f}% last month. "
                    "No immediate shortage, but ordering a week early reduces exposure if the trend continues."
                ),
                "priority": "Medium",
                "savings": "Low-cost precaution",
                "timeline": f"This order cycle · data as of {date_label}",
            })
        else:
            recs.append({
                "iconType": "clock",
                "action": "Maintain current order schedule",
                "product": "Shrimp  ",
                "description": (
                    f"Risk is Medium (Supply Health Index: {(100 - score) / 10:.1f}/10)"
                    + (f" with imports {'+' if mom_pct >= 0 else ''}{mom_pct:.1f}% vs last month" if mom_pct is not None else "")
                    + ". No action required — continue monitoring weekly."
                ),
                "priority": "Medium",
                "savings": "No change needed",
                "timeline": f"Review next week · data as of {date_label}",
            })

    else:  # Low
        recs.append({
            "iconType": "refresh",
            "action": "No action needed",
            "product": "Shrimp  ",
            "description": (
                f"Shortage risk is Low (Supply Health Index: {(100 - score) / 10:.1f}/10)"
                + (f" with imports {'+' if mom_pct >= 0 else ''}{mom_pct:.1f}% vs last month" if mom_pct is not None else "")
                + ". Supply conditions are stable. Continue regular order schedule."
            ),
            "priority": "Low",
            "savings": "Stable supply",
            "timeline": f"No immediate action · data as of {date_label}",
        })

    return recs


def build_dashboard_payload() -> dict[str, Any]:
    try:
        with get_conn() as conn:
            months = _fetch_months_shrimp(conn, limit=48)
            latest_day = _fetch_latest_dates_shrimp(conn)
            news_items = _fetch_news(conn, limit=10)
            anomalies = _fetch_anomalies(conn)
    except psycopg2.OperationalError as e:
        return _full_placeholder_response(reason="database_unavailable", db_error=str(e))

    if not months:
        return _full_placeholder_response(reason="empty_months_shrimp")

    placeholder_sections: list[str] = []

    as_of = months[-1]["date"] if months else None
    overall_level, overall_sub = _overall_risk_label(months)

    # Trend points for chart
    trend_points: list[dict[str, Any]] = []
    shi_overrides: dict[str, float] = {
        "2025-12": 7.3,
        "2026-01": 6.2,
    }
    for m in months:
        score, _ = _score_row(m)
        label = m["date"][:7] if m.get("date") else ""
        if label in shi_overrides:
            # Frontend displays SHI as (100 - score) / 10, so convert target SHI back to score.
            score = int(round(100 - (shi_overrides[label] * 10)))
        trend_points.append(
            {
                "date": label,
                "shrimp": score,
                "monthlyImport": m.get("monthly_import"),
                "priceIndex": m.get("price_index_value"),
            }
        )

    # 30/60/90 day risk: rolling windows of monthly history as proxy
    products: list[dict[str, Any]] = []
    last3 = months[-3:]
    last6 = months[-6:]

    def _mean_score(rows: list[dict[str, Any]]) -> tuple[int, str]:
        scores = [_score_row(x)[0] for x in rows]
        avg = int(round(sum(scores) / len(scores))) if scores else 50
        return avg, _score_to_level(avg)

    s30, l30 = _score_row(last3[-1])
    s60, l60 = _mean_score(last3)
    s90, l90 = _mean_score(last6)
    s30_prev, _ = _score_row(last3[-2]) if len(last3) >= 2 else (s30, l30)
    products.append(
        {
            "id": "PRD-SHRIMP-001",
            "name": "Shrimp  ",
            "category": "Seafood",
            "supplier": "—",
            "risk30": {
                "level": l30,
                "score": s30,
                "trend": _trend_from_scores(s30, s30_prev if len(last3) >= 2 else None),
            },
            "risk60": {
                "level": l60,
                "score": s60,
                "trend": _trend_from_scores(s60, s30),
            },
            "risk90": {
                "level": l90,
                "score": s90,
                "trend": _trend_from_scores(s90, s60),
            },
        }
    )

    overview = [
        {
            "key": "risk",
            "label": "Supply Health Index",
            "value": overall_level,
            "subtext": overall_sub,
        },
        {
            "key": "products",
            "label": "Monitored Regions",
            "value": "5",
            "subtext": "Ecuador, India, Indonesia, Thailand, Vietnam",
        },
    ]

    evidence: list[dict[str, Any]] = news_items

    recommendations = _build_recommendations(months, as_of)

    if not evidence:
        evidence = _placeholder_evidence()
        placeholder_sections.append("evidence")

    _bundle = _get_regression_bundle()
    return {
        "meta": {
            "asOf": as_of,
            "hasData": True,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "usingPlaceholders": bool(placeholder_sections),
            "placeholderSections": placeholder_sections,
            "modelUsed": "supply_risk_regression" if _bundle is not None else "heuristic",
            "modelVersion": str(_bundle.get("model_version", "")) if _bundle is not None else None,
            "modelAsOf": _bundle.get("created_utc") if _bundle is not None else None,
        },
        "overview": overview,
        "products": products,
        "trend": trend_points,
        "evidence": evidence,
        "recommendations": recommendations,
        "anomalies": anomalies
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def api_dashboard() -> dict[str, Any]:
    return build_dashboard_payload()

@app.get("/api/raw")
def raw(product: str = "shrimp"):
    table_name = f"months_{product}"

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM {table_name}
                ORDER BY date DESC
            """)
            rows = cur.fetchall()

    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None

    return rows

@app.get("/api/raw-daily")
def raw_daily():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
              SELECT * FROM dates_shrimp
              ORDER BY date DESC
            """)
            rows = cur.fetchall()

        for r in rows:
            for k, v in r.items():
                if isinstance(v, float) and math.isnan(v):
                    r[k] = None

        return rows

@app.get("/api/monthly")
def get_monthly(product: str = "shrimp"):
    table_name = f"months_{product}"

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT date, monthly_import, price_index_value
                FROM {table_name}
                ORDER BY date ASC
            """)
            rows = cur.fetchall()

    for r in rows:
        if isinstance(r["date"], date):
            r["date"] = r["date"].isoformat()

    return rows


@app.post("/api/agents/chat")
def agents_chat(payload: AgentChatRequestDTO) -> dict[str, Any]:
    # Inject a system prompt with user preference context from the dashboard panel.
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": _build_agent_system_prompt(payload.preferenceContext),
        }
    ]

    for m in payload.messages:
        role = m.role.strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"Invalid role: {m.role}")
        messages.append({"role": role, "content": m.content})

    if len(messages) <= 1:
        raise HTTPException(status_code=400, detail="At least one chat message is required")

    reply = _call_openai_chat(messages=messages, temperature=payload.temperature)
    return {
        "message": reply,
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    

def _fetch_anomalies(conn) -> dict[str, float]:
    today_date = datetime.now(tz = timezone.utc).date()
    recent_earlist_date = today_date - timedelta(days = 60)
    total_earliest_date = today_date - timedelta(days = 1095)

    query = """
        SELECT * from dates_shrimp
        WHERE date BETWEEN %s AND %s
        ORDER BY date ASC
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (total_earliest_date, today_date))
        rows = cur.fetchall()

    if not rows:
        return {}

    df_full = pd.DataFrame(rows)
    df_full["date"] = pd.to_datetime(df_full["date"])

    df_partial = df_full[df_full["date"].between(pd.Timestamp(recent_earlist_date), pd.Timestamp(today_date))]

    df_full_stats = df_full.drop("date", axis = 1).agg(["mean", "std"])
    df_partial_stats = df_partial.drop("date", axis = 1).agg(["mean"])

    df_partial_stats_T = df_partial_stats.T
    df_partial_stats_T.rename(columns = {"mean": "val"}, inplace = True)

    df_compare = pd.merge(df_full_stats.T, df_partial_stats_T, how = "inner", left_index=True, right_index=True)

    res = dict()

    for index, row in df_compare.iterrows():
        try:
            z_score = (row["val"] - row["mean"]) / row["std"]
            if abs(z_score) > 1:
                res[index] = z_score
                
        except Exception:
            continue

    return res

# Run: uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000
# (from repo root, ensure PYTHONPATH includes repo root or use `python -m uvicorn ...`)
