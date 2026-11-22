from config.assets import get_all_assets
from ingestion.company_news import fetch_all_company_news
from ingestion.stock_prices import fetch_all_stock_prices


def main():
    print("=" * 50)
    print("HENGSENG SENTIMENT PRICE PREDICTION AGENT")
    print("Milestone 1: Historical Stock Data Ingestion")
    print("=" * 50)

    assets = get_all_assets()
    print(f"\nTotal assets configured: {len(assets)}")

    successful, failed = fetch_all_stock_prices(assets)

    news_count, news_failed = fetch_all_company_news(assets)

    print("\nIngestion execution completed.")

    if failed == 0:
        print("All stock price datasets downloaded successfully!")
    else:
        print("Some downloads failed. Check the logs above.")

    print(f"News articles collected: {news_count}")
    if news_failed:
        print("Some news searches failed. Check the logs above.")


if __name__ == "__main__":
    main()
