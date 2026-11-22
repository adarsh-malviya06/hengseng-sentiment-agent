import pandas as pd
import numpy as np

from processing.normalize_social import deduplicate_records, find_assets
from sentiment.analyze import engagement_weight, score_frame
from sentiment.aggregate import aggregate_daily
from prediction.models import metrics, train_asset


def test_mixed_timezone_parsing():
    series = pd.to_datetime(pd.Series(["2026-11-01T09:00:00+08:00", "2026-11-02T10:00:00+09:00", "bad"]), format="mixed", errors="coerce", utc=True)
    assert str(series.dtype) == "datetime64[us, UTC]"
    assert series.notna().sum() == 2


def test_asset_mapping_token_and_ambiguity():
    assert find_assets("I like Tencent's earnings", "HKEX") == ["0700.HK"]
    assert not find_assets("the tenth stock", "HKEX")
    assert find_assets("Tencent and Xiaomi", "HKEX") == ["0700.HK", "1810.HK"]
    assert find_assets("삼성전자 실적", "KRX") == ["005930.KS"]


def test_duplicate_removal_uses_id_then_url_then_text_timestamp():
    records = pd.DataFrame([
        {"source": "x", "post_id": "1", "url": None, "text": "a", "created_at_utc": "t1"},
        {"source": "x", "post_id": "1", "url": None, "text": "different", "created_at_utc": "t2"},
        {"source": "x", "post_id": None, "url": "u", "text": "b", "created_at_utc": "t3"},
        {"source": "x", "post_id": None, "url": "u", "text": "c", "created_at_utc": "t4"},
        {"source": "y", "post_id": None, "url": None, "text": "a", "created_at_utc": "t1"},
    ])
    assert len(deduplicate_records(records)) == 3


def test_missing_engagement_is_safe():
    assert engagement_weight({"likes": None, "views": np.nan}) == 1.0
    assert engagement_weight({"likes": 3, "views": 8}) == 3.4641016151377544


def test_sentiment_skips_empty_and_irrelevant():
    df = pd.DataFrame([{"ticker": "0700.HK", "text": "Tencent prospects improve", "likes": 1}, {"ticker": None, "text": "some text", "likes": None}, {"ticker": "0700.HK", "text": "", "likes": None}])
    result = score_frame(df, classifier=lambda texts: [[0.05, 0.05, 0.1, 0.3, 0.5] for _ in texts])
    assert result.sentiment_score.notna().sum() == 1
    assert result.sentiment_label.iloc[0] == "very_positive"


def test_daily_aggregation_uses_exchange_timezone():
    data = pd.DataFrame([{"exchange": "HKEX", "ticker": "0700.HK", "asset_name": "Tencent", "source": "LIHKG", "created_at_utc": "2026-11-01T17:00:00Z", "sentiment_score": 0.5, "sentiment_label": "positive", "engagement_weight": 2}])
    result = aggregate_daily(data)
    assert str(result.trading_date.iloc[0]) == "2026-11-02"
    assert result.total_mentions.iloc[0] == 1
    assert result.weighted_sentiment.iloc[0] == 0.5


def test_metrics_and_directional_accuracy():
    result = metrics([101, 99], [102, 98], [100, 100])
    assert result["mae"] == 1
    assert result["directional_accuracy_pct"] == 100
    assert result["mape_pct"] > 0
    naive = metrics([101, 99], [100, 100], [100, 100])
    assert naive["directional_accuracy_pct"] == 0


def test_price_target_is_next_row_and_latest_row_retained():
    dates = pd.bdate_range("2025-01-01", periods=45)
    close = np.arange(100, 145, dtype=float)
    raw = pd.DataFrame({"date": dates, "ticker": "0700.HK", "asset_name": "Tencent", "exchange": "HKEX", "Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000})
    from processing.price_features import generate_price_features
    result = generate_price_features(raw)
    assert len(result) > 0
    assert pd.isna(result.target_next_close.iloc[-1])
    labeled = result.dropna(subset=["target_next_close"])
    assert labeled.target_next_close.iloc[0] == labeled.Close.iloc[0] + 1


def test_chronological_split_and_no_target_feature():
    dates = pd.bdate_range("2025-01-01", periods=120)
    close = np.linspace(10, 20, len(dates))
    data = pd.DataFrame({"date": dates, "ticker": "0700.HK", "asset_name": "Tencent", "exchange": "HKEX", "Open": close, "High": close+1, "Low": close-1, "Close": close, "Volume": 1000, "daily_return": 0.01, "momentum_5d": 0.02, "momentum_10d": 0.03, "sma_5": close, "sma_10": close, "sma_20": close, "volatility_5d": 0.01, "volatility_10d": 0.01, "daily_range": 0.1, "volume_change": 0.0, "target_next_close": np.r_[close[1:], np.nan], "next_trading_date": list(dates[1:]) + [pd.NaT]})
    from sklearn.dummy import DummyRegressor
    result = train_asset(data, models={"Dummy": DummyRegressor(strategy="mean")})
    assert result["status"] == "ok"
    assert all(group.date.is_monotonic_increasing for _, group in result["predictions"].groupby("model"))
    assert "target_next_close" not in __import__("prediction.models", fromlist=["FEATURES"]).FEATURES
    naive_rows = result["predictions"].query("model == 'Naive (today close)'")
    assert np.allclose(naive_rows.predicted_close, naive_rows.prior_close)
    assert np.allclose(naive_rows.predicted_return_pct, 0)
    assert np.allclose(naive_rows.actual_return_pct, (naive_rows.actual_next_close / naive_rows.prior_close - 1) * 100)
    assert result["forecast"]
    selected = [row["model"] for row in result["forecast"] if row["selected_for_prediction"]]
    assert selected == ["Naive (today close)"]


def test_extreme_forecast_is_flagged_not_clipped():
    dates = pd.bdate_range("2025-01-01", periods=120)
    close = np.linspace(10, 20, len(dates))
    data = pd.DataFrame({"date": dates, "ticker": "0700.HK", "asset_name": "Tencent", "exchange": "HKEX", "Open": close, "High": close+0.1, "Low": close-0.1, "Close": close, "Volume": 1000, "daily_return": 0.01, "momentum_5d": 0.02, "momentum_10d": 0.03, "sma_5": close, "sma_10": close, "sma_20": close, "volatility_5d": 0.01, "volatility_10d": 0.01, "daily_range": 0.01, "volume_change": 0.0, "target_next_close": np.r_[close[1:], np.nan], "next_trading_date": list(dates[1:]) + [pd.NaT]})
    from sklearn.dummy import DummyRegressor
    result = train_asset(data, models={"Extreme test estimator": DummyRegressor(strategy="constant", constant=100)})
    extreme = next(row for row in result["forecast"] if row["model"] == "Extreme test estimator")
    naive = next(row for row in result["forecast"] if row["model"] == "Naive (today close)")
    assert extreme["predicted_return_pct"] > 20
    assert extreme["prediction_warning"] == "outside_historical_daily_return_range"
    assert naive["selected_for_prediction"]
    assert extreme["predicted_next_close"] == 100  # raw forecast is disclosed, not clipped


def test_dashboard_missing_data_loader_returns_empty_frame():
    from dashboard.data import load_csv
    assert load_csv("a-file-that-does-not-exist.csv").empty
