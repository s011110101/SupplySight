# tests/test_shrimp_ingest.py
import pandas as pd
from services.census.ingest_shrimp import clean

def test_clean_basic():
    df = pd.DataFrame({
        "MONTH": ["2018-02", "2018-03", "2018-03"],
        "I_COMMODITY": ["030617", "030617", "030617"],
        "I_COMMODITY_SDESC": ["Frozen shrimp", "Frozen shrimp", "Frozen shrimp"],
        "GEN_VAL_MO": ["1000", "2000", "2000"],
        "VES_WGT_MO": ["100", "200", "200"],
        "CNT_WGT_MO": ["50", "100", "100"],
        "AIR_WGT_MO": ["10", "20", "20"],
    })
    out = clean(df)
    assert out["GEN_VAL_MO"].dtype.kind in ("i","f")
    assert out["VES_WGT_MO"].dtype.kind in ("i","f")
    assert len(out) == 2


def test_run_months_back(monkeypatch, tmp_path):
    # simulate fetch_with_fallback returning fixed data to avoid network
    from services.census.ingest_shrimp import run

    fake_df = pd.DataFrame({
        "I_COMMODITY": ["030617"],
        "I_COMMODITY_SDESC": ["desc"],
        "GEN_VAL_MO": [123],
        "VES_WGT_MO": [1],
        "CNT_WGT_MO": [2],
        "AIR_WGT_MO": [3],
        "MONTH": ["2025-01"],
    })
    monkeypatch.setattr("services.census.ingest_shrimp.fetch_with_fallback", lambda api_key, t1, t2: fake_df)
    monkeypatch.setenv("CENSUS_API_KEY", "key")
    # clear any existing output files so merge starts empty
    from services.census.ingest_shrimp import OUT_CSV, RAW_DIR
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    for f in RAW_DIR.glob("*.csv"):
        f.unlink()

    res = run(months_back=1)
    assert isinstance(res, dict)
    assert res["rows_total"] == 1


def test_feature_engineering(monkeypatch, tmp_path):
    # prepare a tiny input file
    df = pd.DataFrame({
        "I_COMMODITY": ["A", "A"],
        "I_COMMODITY_SDESC": ["foo", "foo"],
        "GEN_VAL_MO": [100, 200],
        "VES_WGT_MO": [10, 20],
        "CNT_WGT_MO": [9, 18],
        "AIR_WGT_MO": [1, 2],
        "MONTH": ["2020-01", "2020-02"],
    })
    in_path = tmp_path / "shrimp_imports.csv"
    out_path = tmp_path / "shrimp_features.csv"
    df.to_csv(in_path, index=False)

    # patch constants to point to tmp paths
    import services.census.feature_engineering as fe
    monkeypatch.setattr(fe, "IN_CSV", in_path)
    monkeypatch.setattr(fe, "OUT_CSV", out_path)

    fe.main()
    assert out_path.exists()
    out = pd.read_csv(out_path)
    assert "total_weight_mo" in out.columns
    assert len(out) == 2


    