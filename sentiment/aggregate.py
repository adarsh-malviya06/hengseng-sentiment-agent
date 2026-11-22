"""Exchange-local daily social sentiment aggregates."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TZ = {"HKEX": "Asia/Hong_Kong", "KRX": "Asia/Seoul"}


def aggregate_daily(df):
    data = df.copy()
    data["created_at_utc"] = pd.to_datetime(data["created_at_utc"], format="mixed", errors="coerce", utc=True)
    data = data[data["sentiment_score"].notna() & data["ticker"].notna()].copy()
    local_dates = []
    for _, row in data.iterrows():
        ts = row["created_at_utc"]
        local_dates.append(ts.tz_convert(TZ[row["exchange"]]).date() if pd.notna(ts) else pd.NaT)
    data["trading_date"] = local_dates
    data["weighted_component"] = data["sentiment_score"] * data["engagement_weight"].fillna(1)
    data["positive_mentions"] = data["sentiment_label"].isin(["positive", "very_positive"]).astype(int)
    data["negative_mentions"] = data["sentiment_label"].isin(["negative", "very_negative"]).astype(int)
    data["neutral_mentions"] = data["sentiment_label"].eq("neutral").astype(int)
    keys = ["exchange", "ticker", "asset_name", "trading_date", "source"]
    result = data.groupby(keys, dropna=False).agg(
        total_mentions=("sentiment_score", "size"), positive_mentions=("positive_mentions", "sum"),
        negative_mentions=("negative_mentions", "sum"), neutral_mentions=("neutral_mentions", "sum"),
        average_sentiment=("sentiment_score", "mean"),
        weighted_sentiment=("weighted_component", "sum"), weight_sum=("engagement_weight", "sum"),
    ).reset_index()
    result["weighted_sentiment"] = result["weighted_sentiment"] / result["weight_sum"].replace(0, pd.NA)
    return result.drop(columns="weight_sum").sort_values(keys[:4])


def run_aggregation(input_file=None):
    path = Path(input_file) if input_file else ROOT / "data" / "processed" / "social" / "recent_social_sentiment.csv"
    result = aggregate_daily(pd.read_csv(path))
    out = path.with_name("daily_social_sentiment.csv")
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved daily sentiment: {out}")
    return result


def aggregate_news(df):
    data = df.copy()
    from processing.normalize_social import ASSETS

    if data.empty:
        return pd.DataFrame(columns=[
            "exchange", "ticker", "asset_name", "trading_date", "source",
            "total_mentions", "positive_mentions", "negative_mentions",
            "neutral_mentions", "average_sentiment", "weighted_sentiment",
        ])

    # Only known tickers paired with their configured exchange are aggregated.
    if "ticker" not in data:
        data["ticker"] = pd.NA
    if "exchange" not in data:
        data["exchange"] = pd.NA
    data = data[data["ticker"].isin(ASSETS)].copy()
    expected_exchange = data["ticker"].map(lambda ticker: ASSETS.get(ticker, (None, None, None))[1])
    data = data[data["exchange"].eq(expected_exchange) & data["exchange"].isin(TZ)].copy()
    if data.empty:
        return pd.DataFrame(columns=[
            "exchange", "ticker", "asset_name", "trading_date", "source",
            "total_mentions", "positive_mentions", "negative_mentions",
            "neutral_mentions", "average_sentiment", "weighted_sentiment",
        ])

    if "engagement_weight" not in data:
        data["engagement_weight"] = 1.0
    if "source" not in data:
        data["source"] = "Exa"
    if "asset_name" not in data:
        data["asset_name"] = data["ticker"].map(lambda ticker: ASSETS[ticker][0])
    data["published_date"] = pd.to_datetime(data["published_date"], format="mixed", errors="coerce", utc=True)
    data = data[data["sentiment_score"].notna() & data["ticker"].notna()].copy()
    data["trading_date"] = [ts.tz_convert(TZ[exchange]).date() if pd.notna(ts) else pd.NaT for ts, exchange in zip(data.published_date, data.exchange)]
    data["source"] = data["source"].fillna("Exa")
    data["weighted_component"] = data.sentiment_score * data.engagement_weight.fillna(1)
    data["positive_mentions"] = data.sentiment_label.isin(["positive", "very_positive"]).astype(int)
    data["negative_mentions"] = data.sentiment_label.isin(["negative", "very_negative"]).astype(int)
    data["neutral_mentions"] = data.sentiment_label.eq("neutral").astype(int)
    keys = ["exchange", "ticker", "asset_name", "trading_date", "source"]
    result = data.groupby(keys, dropna=False).agg(total_mentions=("sentiment_score", "size"), positive_mentions=("positive_mentions", "sum"), negative_mentions=("negative_mentions", "sum"), neutral_mentions=("neutral_mentions", "sum"), average_sentiment=("sentiment_score", "mean"), weighted_sentiment=("weighted_component", "sum"), weight_sum=("engagement_weight", "sum")).reset_index()
    result["weighted_sentiment"] /= result["weight_sum"].replace(0, pd.NA)
    return result.drop(columns="weight_sum")


def run_news_aggregation():
    path = ROOT / "data" / "processed" / "news" / "news_sentiment.csv"
    result = aggregate_news(pd.read_csv(path))
    out = path.with_name("daily_news_sentiment.csv")
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved daily news sentiment: {out}")
    return result


if __name__ == "__main__":
    run_aggregation()
