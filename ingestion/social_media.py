
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from apify_client import ApifyClient
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_SOCIAL_DIR = BASE_DIR / "data" / "raw" / "social"

load_dotenv(BASE_DIR / ".env")

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

LIHKG_ACTOR_ID = "claude_code_reviewer/lihkg-scraper-en"
NAVER_ACTOR_ID = "hgservices/naver-stock-discussion-scraper"


def get_apify_client():
    """Create an authenticated Apify client."""

    if not APIFY_TOKEN:
        raise ValueError(
            "APIFY_API_TOKEN not found. Check your .env file."
        )

    return ApifyClient(APIFY_TOKEN)


def save_json(data, filename):
    """Save scraped records as JSON."""

    RAW_SOCIAL_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RAW_SOCIAL_DIR / filename

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Saved: {output_file}")


def scrape_lihkg_test(client):
    """Test LIHKG with a small Tencent-related search."""

    print("\n========== LIHKG TEST ==========")

    run_input = {
        "mode": "search",
        "keywords": ["Tencent", "騰訊"],
        "order": "desc_reply_time",
        "dateFrom": "10d",
        "maxItems": 10
    }

    run = client.actor(LIHKG_ACTOR_ID).call(
        run_input=run_input
    )

    if not run:
        raise RuntimeError("LIHKG actor did not return a run.")

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("LIHKG run has no dataset ID.")

    records = list(
        client.dataset(dataset_id).iterate_items()
    )

    save_json(records, "lihkg_test.json")

    print(f"LIHKG records collected: {len(records)}")

    return records


def scrape_naver_test(client):
    """Test Naver with Samsung Electronics discussion posts."""

    print("\n========== NAVER TEST ==========")

    # Naver expects the date in Korea local time.
    today_korea = datetime.now(
        ZoneInfo("Asia/Seoul")
    ).date()

    date_from = (
        today_korea - timedelta(days=9)
    ).isoformat()

    run_input = {
        "stocks": ["005930"],
        "dateFrom": date_from,
        "maxResults": 10
    }

    run = client.actor(NAVER_ACTOR_ID).call(
        run_input=run_input
    )

    if not run:
        raise RuntimeError("Naver actor did not return a run.")

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("Naver run has no dataset ID.")

    records = list(
        client.dataset(dataset_id).iterate_items()
    )

    save_json(records, "naver_test.json")

    print(f"Naver records collected: {len(records)}")

    return records


def main():
    print("=" * 50)
    print("HENGSENG AGENT - SOCIAL MEDIA INGESTION TEST")
    print("=" * 50)

    try:
        client = get_apify_client()

        lihkg_records = scrape_lihkg_test(client)
        naver_records = scrape_naver_test(client)

        print("\n========== TEST SUMMARY ==========")
        print(f"LIHKG records: {len(lihkg_records)}")
        print(f"Naver records: {len(naver_records)}")
        print("=================================")

        print("\nSocial-media test completed.")

    except Exception as error:
        print(f"\n[ERROR] {error}")
        raise


if __name__ == "__main__":
    main()
