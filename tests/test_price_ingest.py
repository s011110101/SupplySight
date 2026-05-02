import pandas as pd
import pytest

from services.price.ingest_fao_price_index import (
    OUTPUT_COLUMNS,
    clean,
    extract_shrimp_series,
    find_newest_fao_csv,
    parse_csv_links,
)


def test_extract_shrimp_series(tmp_path):
    csv_path = tmp_path / "fao.csv"
    csv_path.write_text(
        """FAO Fish Price Index,,,,,,
Base period: 2014-16=100. ,,,,,,
Nominal Indices,,,,,,
Date,FAO Fish Price Index,Tuna,Pelagic (excl. tuna),Salmon,Shrimp,Whitefish
Jan-90,69.1,51.5,48.8,81.6,83.9,56.1
Feb-90,68.1,53.4,43.1,80.1,83.2,56.1
""",
        encoding="utf-8",
    )

    df = extract_shrimp_series(csv_path)

    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "date"] == "1990-01-01"
    assert df.loc[0, "commodity"] == "Shrimp"
    assert df.loc[0, "source"] == "FAO_FishPriceIndex"


def test_extract_shrimp_series_missing_columns(tmp_path):
    csv_path = tmp_path / "bad_fao.csv"
    csv_path.write_text(
        """FAO Fish Price Index,,,,,,
Base period: 2014-16=100. ,,,,,,
Nominal Indices,,,,,,
Date,Tuna
Jan-90,51.5
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        extract_shrimp_series(csv_path)


def test_clean_deduplicates_by_date():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "commodity": ["Shrimp", "Shrimp", "Shrimp"],
            "value": [1, 2, 3],
            "source": ["FAO", "FAO", "FAO"],
            "source_file": ["a.csv", "b.csv", "c.csv"],
            "ingested_at": ["t1", "t2", "t3"],
        }
    )

    out = clean(df)

    assert list(out["date"]) == ["2024-01-01", "2024-02-01"]
    assert list(out["value"]) == [2, 3]


def test_parse_csv_links_uses_dir_url():
    html = """
    <html><body>
      <table>
        <tr><td><a href="FAO_fish_price_index_Jan2026.csv">Jan</a></td><td>10 KB</td><td>2026-02-01 12:30</td></tr>
      </table>
    </body></html>
    """

    links = parse_csv_links(html, "https://example.com/custom-dir/")

    assert len(links) == 1
    assert links[0]["url"] == "https://example.com/custom-dir/FAO_fish_price_index_Jan2026.csv"


def test_find_newest_fao_csv_picks_latest(monkeypatch):
    html = """
    <html><body><table>
      <tr><td><a href="FAO_fish_price_index_Feb2026.csv">Feb</a></td><td>10 KB</td><td>2026-02-15 09:00</td></tr>
      <tr><td><a href="FAO_fish_price_index_Jan2026.csv">Jan</a></td><td>10 KB</td><td>2026-01-15 09:00</td></tr>
    </table></body></html>
    """
    monkeypatch.setattr(
        "services.price.ingest_fao_price_index.fetch_directory_listing",
        lambda dir_url: html,
    )

    name, url = find_newest_fao_csv("https://example.com/fao/")

    assert name == "FAO_fish_price_index_Feb2026.csv"
    assert url == "https://example.com/fao/FAO_fish_price_index_Feb2026.csv"


def test_find_newest_fao_csv_falls_back_to_filename(monkeypatch):
    html = """
    <html><body><table>
      <tr><td><a href="FAO_fish_price_index_Dec2025.csv">Dec</a></td><td>10 KB</td><td></td></tr>
      <tr><td><a href="FAO_fish_price_index_Jan2026.csv">Jan</a></td><td>10 KB</td><td></td></tr>
    </table></body></html>
    """
    monkeypatch.setattr(
        "services.price.ingest_fao_price_index.fetch_directory_listing",
        lambda dir_url: html,
    )

    name, _ = find_newest_fao_csv("https://example.com/fao/")

    assert name == "FAO_fish_price_index_Jan2026.csv"
