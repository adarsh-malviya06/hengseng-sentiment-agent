"""Multilingual transformer sentiment and transparent engagement weighting."""
from pathlib import Path
import math

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]


def engagement_weight(row):
    """sqrt(1 + likes + dislikes + views + reposts), capped at 10."""
    values = [row.get(k) for k in ("likes", "dislikes", "views", "reposts")]
    total = sum(max(0, float(v)) for v in values if pd.notna(v))
    return min(10.0, math.sqrt(1.0 + total))


def classify_texts(texts, classifier=None):
    """Return model probabilities; pass a classifier in tests to avoid downloads."""
    if classifier is None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to run multilingual sentiment (see requirements.txt).") from exc
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

        def classifier(batch):
            encoded = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().tolist()
            return probs

    return classifier(texts)


def score_frame(df, classifier=None):
    out = df.copy()
    out["sentiment_label"] = None
    out["positive_score"] = pd.NA
    out["negative_score"] = pd.NA
    out["neutral_score"] = pd.NA
    out["sentiment_confidence"] = pd.NA
    out["sentiment_score"] = pd.NA
    out["engagement_weight"] = out.apply(engagement_weight, axis=1)
    mask = out["ticker"].notna() & out["text"].fillna("").str.strip().ne("")
    if not mask.any():
        return out
    probs = classify_texts(out.loc[mask, "text"].tolist(), classifier)
    for idx, p in zip(out.index[mask], probs):
        p = [float(x) for x in p]
        label_i = max(range(len(p)), key=p.__getitem__)
        out.at[idx, "sentiment_label"] = LABELS[label_i]
        out.at[idx, "negative_score"] = p[0] + p[1]
        out.at[idx, "neutral_score"] = p[2]
        out.at[idx, "positive_score"] = p[3] + p[4]
        out.at[idx, "sentiment_confidence"] = p[label_i]
        out.at[idx, "sentiment_score"] = sum((i - 2) * prob for i, prob in enumerate(p)) / 2
    return out


def run_sentiment(input_file=None, recent=True):
    path = Path(input_file) if input_file else ROOT / "data" / "processed" / "social" / ("recent_social_posts.csv" if recent else "all_social_posts.csv")
    df = pd.read_csv(path)
    scored = score_frame(df)
    target = path.with_name("recent_social_sentiment.csv" if recent else "all_social_sentiment.csv")
    scored.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"Scored {scored.sentiment_score.notna().sum()} relevant non-empty records: {target}")
    return scored


def run_news_sentiment():
    """Score Exa articles separately from social records."""
    source_dir = ROOT / "data" / "raw" / "company_news"
    files = list(source_dir.glob("*.csv")) if source_dir.exists() else []
    if not files:
        raise FileNotFoundError("No Exa news CSVs found; run main.py with EXA_API_KEY configured.")
    articles = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    # Accept older saved Exa files while keeping the current output schema.
    articles = articles.rename(columns={"company": "asset_name", "highlights": "summary"})
    for column in ("ticker", "exchange", "published_date", "title", "summary", "url", "source"):
        if column not in articles:
            articles[column] = pd.NA
    if "url" in articles:
        has_url = articles["url"].notna() & articles["url"].astype(str).str.strip().ne("")
        with_url = articles.loc[has_url].drop_duplicates("url", keep="last")
        without_url = articles.loc[~has_url].drop_duplicates(["title", "published_date"], keep="last")
        articles = pd.concat([with_url, without_url], ignore_index=True)
    articles["text"] = (articles["title"].fillna("").astype(str) + ". " + articles["summary"].fillna("").astype(str)).str.strip(" .")
    from processing.normalize_social import find_assets
    articles["matched_tickers"] = [
        find_assets(text, exchange if isinstance(exchange, str) else None)
        for text, exchange in zip(articles.text, articles.exchange)
    ]
    articles["relevance_status"] = [
        "relevant_single_asset" if len(matches) == 1 and matches[0] == ticker
        else "ticker_mismatch" if len(matches) == 1 and matches[0] != ticker
        else "ambiguous_asset" if len(matches) > 1
        else "unmatched"
        for matches, ticker in zip(articles.matched_tickers, articles.ticker)
    ]
    articles["ticker"] = [
        matches[0] if len(matches) == 1 and matches[0] == ticker else pd.NA
        for matches, ticker in zip(articles.matched_tickers, articles.ticker)
    ]
    from processing.normalize_social import ASSETS
    articles["asset_name"] = articles["ticker"].map(
        lambda ticker: ASSETS.get(ticker, (None, None, None))[0]
        if isinstance(ticker, str)
        else None
    )
    scored = score_frame(articles)
    out = ROOT / "data" / "processed" / "news" / "news_sentiment.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Scored {scored.sentiment_score.notna().sum()} news articles: {out}")
    return scored


if __name__ == "__main__":
    run_sentiment()
