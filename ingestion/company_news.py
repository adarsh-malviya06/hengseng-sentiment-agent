"""Fetch and store recent company news with Exa search."""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from exa_py import Exa


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "company_news"


def fetch_company_news(asset):
    """Fetch recent news for one configured asset and save it to CSV."""
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is missing. Add it to the project .env file.")

    ticker = asset["ticker"]
    name = asset["name"]
    exchange = asset["exchange"]
    query = f"Latest news and company announcements about {name} ({ticker})"
    searched_at = datetime.now(timezone.utc).isoformat()
    published_after = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()

    exa = Exa(api_key=api_key)
    response = exa.search(
        query,
        objective=(
            f"Monitor recent news about {name} ({ticker}) for a stock sentiment pipeline. "
            "Prioritize relevant news articles and company announcements; capture the "
            "publication date, title, URL, and key highlights."
        ),
        contents={"highlights": True},
        num_results=10,
        start_published_date=published_after,
    )

    articles = []
    for item in response.results:
        highlights = getattr(item, "highlights", None) or []
        articles.append(
            {
                "ticker": ticker,
                "asset_name": name,
                "exchange": exchange,
                "published_date": getattr(item, "published_date", None),
                "title": getattr(item, "title", None),
                "url": getattr(item, "url", None),
                "summary": " ".join(highlights)
                if isinstance(highlights, list)
                else str(highlights),
                "source": getattr(item, "author", None) or getattr(item, "source", None) or (urlparse(getattr(item, "url", "")).netloc or None),
                "search_timestamp_utc": searched_at,
            }
        )

    if not articles:
        print(f"[WARNING] No news found for {name} ({ticker})")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{ticker.replace('.', '_')}.csv"
    current = pd.DataFrame(articles)

    if output_file.exists():
        previous = pd.read_csv(output_file)
        previous = previous.rename(columns={"company": "asset_name", "highlights": "summary"})
        current = pd.concat([previous, current], ignore_index=True)

    current["_dedupe_key"] = [
        f"url:{url}" if pd.notna(url) and str(url).strip() else f"title:{title}|date:{published}"
        for url, title, published in zip(current.get("url"), current.get("title"), current.get("published_date"))
    ]
    current = current.drop_duplicates(subset=["_dedupe_key"], keep="last").drop(columns="_dedupe_key")
    current.to_csv(output_file, index=False)
    print(f"[SUCCESS] Saved {len(articles)} news results for {name}: {output_file}")
    return len(articles)


def fetch_all_company_news(assets):
    """Fetch recent news for all configured assets."""
    total_articles = 0
    failed = 0

    for asset in assets:
        try:
            total_articles += fetch_company_news(asset)
        except Exception as error:
            failed += 1
            print(f"[ERROR] News fetch failed for {asset['ticker']}: {error}")

    print("\n========== NEWS MONITORING SUMMARY ==========")
    print(f"Articles fetched: {total_articles}")
    print(f"Assets failed: {failed}")
    print("=============================================")
    return total_articles, failed
