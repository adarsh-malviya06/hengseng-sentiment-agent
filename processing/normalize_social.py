"""Normalize scraped social records without changing their raw source files."""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOCIAL_DIR = ROOT / "data" / "raw" / "social"
OUTPUT_DIR = ROOT / "data" / "processed" / "social"
WINDOW_DAYS = int(os.getenv("SOCIAL_WINDOW_DAYS", "10"))

# Unicode aliases are intentional; ASCII aliases use token boundaries below.
ASSETS = {
    "0700.HK": ("Tencent", "HKEX", ["tencent", "騰訊", "腾讯"]),
    "9988.HK": ("Alibaba", "HKEX", ["alibaba", "阿里巴巴"]),
    "3690.HK": ("Meituan", "HKEX", ["meituan", "美團", "美团"]),
    "1810.HK": ("Xiaomi", "HKEX", ["xiaomi", "小米"]),
    "0005.HK": ("HSBC", "HKEX", ["hsbc", "匯豐", "汇丰"]),
    "005930.KS": ("Samsung Electronics", "KRX", ["samsung electronics", "samsung", "삼성전자", "005930"]),
    "000660.KS": ("SK Hynix", "KRX", ["sk hynix", "sk하이닉스", "하이닉스", "000660"]),
    "005380.KS": ("Hyundai Motor", "KRX", ["hyundai motor", "hyundai", "현대자동차", "현대차", "005380"]),
    "066570.KS": ("LG Electronics", "KRX", ["lg electronics", "lg전자", "066570"]),
    "035420.KS": ("NAVER", "KRX", ["naver", "네이버", "035420"]),
}


def load_json(path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unexpected JSON structure in {path}")


def find_assets(text, exchange=None):
    """Find explicit whole-token/company aliases; return all matching tickers."""
    text = str(text or "").casefold()
    matches = set()
    for ticker, (_, market, aliases) in ASSETS.items():
        if exchange and market != exchange:
            continue
        for alias in aliases:
            alias = alias.casefold()
            if alias.isascii() and re.fullmatch(r"[a-z0-9 .-]+", alias):
                pattern = r"(?<![a-z0-9])" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
                found = re.search(pattern, text)
            else:
                found = alias in text
            if found:
                matches.add(ticker)
                break
    return sorted(matches)


def deduplicate_records(df):
    """Deduplicate by source ID, then URL, then text plus timestamp."""
    out = df.copy()
    out["_dedupe_key"] = [
        f"id:{row.post_id}" if pd.notna(row.post_id) else
        f"url:{row.url}" if pd.notna(row.url) and str(row.url).strip() else
        f"text:{row.text}|date:{row.created_at_utc}"
        for row in out.itertuples()
    ]
    return out.drop_duplicates(subset=["source", "_dedupe_key"], keep="first").drop(columns="_dedupe_key").reset_index(drop=True)


def _base(source, exchange, post_id, text, date, author=None, url=None):
    matches = find_assets(text, exchange)
    ticker = matches[0] if len(matches) == 1 else None
    name = ASSETS[ticker][0] if ticker else None
    return {
        "source": source, "exchange": exchange, "ticker": ticker, "asset_name": name,
        "post_id": post_id, "text": text, "created_at_utc": date, "author": author,
        "author_followers": None, "likes": None, "dislikes": None, "reposts": None,
        "views": None, "url": url, "channel_name": None, "channel_username": None,
        "channel_subscribers": None, "channel_verified": None,
        "relevance_status": "ambiguous_asset" if len(matches) > 1 else ("company_match_review" if ticker else "unmatched"),
    }


def normalize_records():
    rows = []
    for path in sorted(SOCIAL_DIR.glob("*.json")):
        records = load_json(path)
        for item in records:
            if path.name.startswith("lihkg"):
                text = " ".join(str(item.get(k) or "") for k in ("title", "content", "own_text")).strip()
                row = _base("LIHKG", "HKEX", item.get("thread_id"), text, item.get("created_at") or item.get("last_reply_at"), item.get("author"), item.get("thread_url"))
                row["likes"], row["dislikes"] = item.get("like_count"), item.get("dislike_count")
                if row["ticker"]:
                    row["relevance_status"] = "investor_forum_company_match"
            elif path.name.startswith("naver"):
                text = " ".join(str(item.get(k) or "") for k in ("title", "text")).strip()
                code = str(item.get("stockCode") or "").strip()
                ticker = f"{code}.KS" if code else None
                if ticker not in ASSETS:
                    ticker = None
                row = _base("Naver Finance", "KRX", item.get("postId"), text, item.get("writtenAt"), item.get("author"), item.get("url"))
                # The Naver stock board gives explicit asset context; preserve it.
                if ticker:
                    row.update(ticker=ticker, asset_name=ASSETS[ticker][0], relevance_status="investor_forum_company_match")
                row.update(views=item.get("views"), likes=item.get("likes"), dislikes=item.get("dislikes"))
            elif path.name.startswith("telegram"):
                preview = item.get("linkPreview") or {}
                text = " ".join(str(v) for v in (item.get("text"), preview.get("title"), preview.get("description")) if v).strip()
                # Telegram keyword collection is KRX-oriented; require an explicit company alias.
                row = _base("Telegram", "KRX", item.get("messageId"), text, item.get("date"), item.get("authorSignature"), item.get("url"))
                row.update(views=item.get("views"), likes=item.get("reactionsTotal"), channel_name=item.get("channelTitle"), channel_username=item.get("channelUsername"), channel_subscribers=item.get("channelSubscribers"), channel_verified=item.get("channelVerified"))
            else:
                continue
            row["source_file"] = path.name
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # `mixed` handles ISO strings with different offsets in one Series; utc=True
    # normalizes aware values and treats naive source times consistently as UTC.
    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], format="mixed", errors="coerce", utc=True)
    df = deduplicate_records(df)
    return df.sort_values("created_at_utc", na_position="last").reset_index(drop=True)


def write_normalized(window_days=WINDOW_DAYS, collected_at=None):
    """Write complete normalized history and a separate recent-window dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = normalize_records()
    if df.empty:
        raise FileNotFoundError(f"No social JSON records found in {SOCIAL_DIR}")
    now = pd.Timestamp(collected_at or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    cutoff = now - pd.Timedelta(days=window_days)
    recent = df[df["created_at_utc"].between(cutoff, now, inclusive="both")].copy()
    df.to_csv(OUTPUT_DIR / "all_social_posts.csv", index=False, encoding="utf-8-sig")
    recent.to_csv(OUTPUT_DIR / "recent_social_posts.csv", index=False, encoding="utf-8-sig")
    print(f"Normalized {len(df)} unique posts; valid UTC timestamps: {df.created_at_utc.notna().sum()}; recent ({window_days}d): {len(recent)}")
    print(f"Saved complete and recent datasets under {OUTPUT_DIR}")
    return df, recent


def main():
    write_normalized()


if __name__ == "__main__":
    main()
