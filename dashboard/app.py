"""Streamlit dashboard backed only by project-generated datasets."""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from prediction.models import support_resistance
from dashboard.data import load_csv

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="HengSeng Sentiment Agent", layout="wide")
st.title("HengSeng Sentiment Price Prediction Agent")
st.caption("HKEX and KRX monitoring · research estimates, not investment advice")


prices = load_csv(ROOT / "data/processed/all_assets_processed.csv")
social = load_csv(ROOT / "data/processed/social/recent_social_sentiment.csv")
daily_social = load_csv(ROOT / "data/processed/social/daily_social_sentiment.csv")
news = load_csv(ROOT / "data/processed/news/news_sentiment.csv")
preds = load_csv(ROOT / "data/predictions/latest_predictions.csv")
metrics = load_csv(ROOT / "data/predictions/model_metrics.csv")
page = st.sidebar.radio("Section", ["Overview", "HKEX", "KRX", "Asset Analysis", "Prediction and Evaluation", "Data Sources and Quality"])

if page == "Overview":
    assets = prices.ticker.nunique() if not prices.empty else 0
    cols = st.columns(4)
    cols[0].metric("Assets monitored", assets)
    cols[1].metric("HKEX assets", prices.loc[prices.exchange == "HKEX", "ticker"].nunique() if not prices.empty else 0)
    cols[2].metric("KRX assets", prices.loc[prices.exchange == "KRX", "ticker"].nunique() if not prices.empty else 0)
    cols[3].metric("Recent social mentions", len(social))
    st.info("Sentiment uses only collected recent records. Historical sentiment is unavailable, so backtests are price-only.")
    if not social.empty and "sentiment_score" in social:
        st.metric("Mean recent sentiment", f"{social.sentiment_score.mean():.3f}")
        rank = social.groupby(["ticker", "asset_name"]).sentiment_score.mean().sort_values()
        st.subheader("Recent asset sentiment")
        st.dataframe(rank.rename("mean_sentiment").reset_index(), use_container_width=True)
        st.write("Most positive", ", ".join(rank.tail(3).index.get_level_values("ticker").tolist()) or "No scored assets")
        st.write("Most negative", ", ".join(rank.head(3).index.get_level_values("ticker").tolist()) or "No scored assets")
    if not news.empty and "sentiment_score" in news:
        st.subheader("Recent Exa news sentiment")
        st.dataframe(news.groupby(["ticker", "asset_name"]).sentiment_score.mean().rename("mean_news_sentiment").reset_index(), use_container_width=True)
    if not preds.empty:
        st.subheader("Latest price-only estimates")
        st.dataframe(preds, use_container_width=True)
    warnings = []
    if social.empty: warnings.append("No recent social sentiment file is available.")
    if news.empty: warnings.append("No combined news file has been produced.")
    if preds.empty: warnings.append("No prediction results are available.")
    if warnings: st.warning(" ".join(warnings))
elif page in ("HKEX", "KRX"):
    exchange = page
    st.header(f"{exchange} Dashboard")
    if not prices.empty:
        st.write("Configured assets", ", ".join(sorted(prices.loc[prices.exchange == exchange, "ticker"].unique())))
    table = social[social.exchange == exchange] if not social.empty and "exchange" in social else pd.DataFrame()
    if table.empty: st.info("No scored social records are available for this exchange in the recent window.")
    else: st.dataframe(table.groupby(["ticker", "asset_name", "source"]).agg(mentions=("text", "count"), sentiment=("sentiment_score", "mean")).reset_index(), use_container_width=True)
    if not preds.empty: st.dataframe(preds[preds.exchange == exchange], use_container_width=True)
elif page == "Asset Analysis":
    options = sorted(prices.ticker.dropna().unique()) if not prices.empty else []
    if not options: st.warning("Process price data first.")
    else:
        ticker = st.selectbox("Asset", options)
        p = prices[prices.ticker == ticker].sort_values("date")
        st.plotly_chart(px.line(p, x="date", y="Close", title=f"{ticker} close price"), use_container_width=True)
        sr = support_resistance(p)
        st.write(f"Current close: {sr['current_price']:.2f} · recent trend: {sr['trend']}")
        st.write("Observed swing support:", sr["support"] or "No recent swing lows")
        st.write("Observed swing resistance:", sr["resistance"] or "No recent swing highs")
        s = social[social.ticker == ticker] if not social.empty and "ticker" in social else pd.DataFrame()
        if not s.empty:
            st.plotly_chart(px.scatter(s, x="created_at_utc", y="sentiment_score", color="source", hover_data=["text"]), use_container_width=True)
            st.dataframe(s.tail(30), use_container_width=True)
        else: st.info("No recent scored social posts for this asset.")
        trend = daily_social[daily_social.ticker == ticker] if not daily_social.empty and "ticker" in daily_social else pd.DataFrame()
        if not trend.empty:
            st.plotly_chart(px.line(trend, x="trading_date", y="weighted_sentiment", color="source", title="Daily engagement-weighted social sentiment"), use_container_width=True)
        n = news[news.ticker == ticker] if not news.empty and "ticker" in news else pd.DataFrame()
        if not n.empty:
            st.subheader("Recent relevant news")
            st.dataframe(n[[c for c in ["published_date", "title", "summary", "source", "sentiment_label", "sentiment_score", "url"] if c in n]].head(20), use_container_width=True)
        forecast = preds[preds.ticker == ticker] if not preds.empty else pd.DataFrame()
        if not forecast.empty: st.dataframe(forecast, use_container_width=True)
elif page == "Prediction and Evaluation":
    st.header("Price-only model evaluation")
    st.info("No historical sentiment-enhanced backtest is claimed: the available social history covers only a recent sample.")
    if metrics.empty: st.warning("Run the prediction pipeline to create metrics.")
    else: st.dataframe(metrics, use_container_width=True)
    backtest = load_csv(ROOT / "data/predictions/backtest_predictions.csv")
    if not backtest.empty:
        ticker = st.selectbox("Backtest asset", sorted(backtest.ticker.unique()))
        st.plotly_chart(px.line(backtest[backtest.ticker == ticker], x="date", y=["actual_next_close", "predicted_close"], color="model"), use_container_width=True)
    if not preds.empty: st.dataframe(preds, use_container_width=True)
elif page == "Data Sources and Quality":
    st.header("Data sources and quality")
    counts = []
    if not social.empty:
        counts.extend({"source": k, "records": int(v)} for k, v in social.source.value_counts().items())
    if not news.empty: counts.append({"source": "Exa news", "records": len(news)})
    st.dataframe(pd.DataFrame(counts), use_container_width=True)
    if not social.empty:
        st.write("Newest social timestamp (UTC):", social.get("created_at_utc", pd.Series(dtype=str)).max())
        missing = social.isna().sum().rename("missing_values").reset_index(names="field")
        st.subheader("Missing fields in recent social records")
        st.dataframe(missing[missing.missing_values > 0], use_container_width=True)
    st.markdown("- Social coverage: LIHKG for HKEX; Naver Finance and Telegram for KRX. A second HKEX source is not yet validated.\n- Raw social files are retained separately from normalized and ten-day filtered outputs.\n- Channel subscribers are not treated as author followers.\n- Exa and Apify API keys are never displayed.")
    if not prices.empty: st.write("Price rows:", len(prices), " · latest date:", prices.date.max())
