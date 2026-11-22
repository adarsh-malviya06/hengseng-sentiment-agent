from pathlib import Path

import pandas as pd
import yfinance as yf


# Project root directory.
BASE_DIR = Path(__file__).resolve().parents[1]

# Where downloaded prices will be stored.
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "stock_prices"


def fetch_stock_prices(ticker, name, exchange):
    """Download approximately 2 years of daily historical stock prices."""
    print(f"\nDownloading: {name} ({ticker})")

    try:
        stock = yf.Ticker(ticker)
        history = stock.history(
            period="2y",
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

        if history.empty:
            print(f"[WARNING] No data available for {ticker}")
            return False

        # Preserve the exchange-local trading date.
        history.index.name = "date"
        history = history.reset_index()

        # Standardize the date column.
        date_column = next(
            column
            for column in history.columns
            if str(column).lower() in ("date", "datetime")
        )
        history = history.rename(columns={date_column: "date"})

        history["ticker"] = ticker
        history["asset_name"] = name
        history["exchange"] = exchange

        # Select required columns.
        columns = [
            "date",
            "ticker",
            "asset_name",
            "exchange",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
        history = history[columns]

        # Save CSV.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"{ticker.replace('.', '_')}.csv"
        history.to_csv(output_file, index=False)

        print(f"[SUCCESS] Saved {len(history)} records")
        print(f"File: {output_file}")
        return True

    except Exception as error:
        print(f"[ERROR] {ticker}: {error}")
        return False


def fetch_all_stock_prices(assets):
    """Download historical prices for all configured assets."""
    successful = 0
    failed = 0

    for asset in assets:
        result = fetch_stock_prices(
            ticker=asset["ticker"],
            name=asset["name"],
            exchange=asset["exchange"],
        )

        if result:
            successful += 1
        else:
            failed += 1

    print("\n========== INGESTION SUMMARY ==========")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("=======================================")

    return successful, failed
