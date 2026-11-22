
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "raw" / "stock_prices"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

REQUIRED_COLUMNS = [
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


def process_stock_file(file_path):
    """Clean one stock CSV and generate market features."""

    print(f"\nProcessing: {file_path.name}")

    try:
        df = generate_price_features(pd.read_csv(file_path))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / file_path.name
        df.to_csv(output_file, index=False)
        print(f"[SUCCESS] {len(df)} processed records")
        print(f"Saved: {output_file}")
        return df
    except Exception as error:
        print(f"[ERROR] {file_path.name}: {error}")
        return None


def generate_price_features(df):
    """Create causal OHLCV features and the next-row close target."""

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Parse dates and numeric values. Price dates are exchange-local trading dates.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove invalid and duplicate records.
    df = df.dropna(subset=["date", "Open", "High", "Low", "Close"])

    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("Not enough valid records for feature generation.")

    # Daily market movement.
    df["daily_return"] = df["Close"].pct_change()

    # Price momentum.
    df["momentum_5d"] = df["Close"].pct_change(5)
    df["momentum_10d"] = df["Close"].pct_change(10)

    # Moving averages.
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()

    # Volatility based on daily returns.
    df["volatility_5d"] = df["daily_return"].rolling(5).std()

    df["volatility_10d"] = df["daily_return"].rolling(10).std()

    # Daily trading range.
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]

    # Volume movement.
    df["volume_change"] = df["Volume"].pct_change()

    # Target: next available trading session's closing price.
    df["next_trading_date"] = df["date"].shift(-1)
    df["target_next_close"] = df["Close"].shift(-1)

    # Target: next session's percentage price movement.
    df["target_next_return"] = (df["target_next_close"] / df["Close"]) - 1

    # Retain the latest row without its next-session target for forecasting.
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    feature_columns = ["daily_return", "momentum_5d", "momentum_10d", "sma_5", "sma_10", "sma_20", "volatility_5d", "volatility_10d", "daily_range", "volume_change"]
    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable records remain after processing.")
    return df


def process_all_stock_files():
    """Process all downloaded stock-price CSV files."""

    if not INPUT_DIR.exists():
        print("[ERROR] Raw stock-price directory not found.")
        return

    files = list(INPUT_DIR.glob("*.csv"))

    if not files:
        print("[ERROR] No stock-price CSV files found.")
        return

    processed_data = []
    successful = 0
    failed = 0

    for file_path in files:
        result = process_stock_file(file_path)

        if result is not None:
            processed_data.append(result)
            successful += 1
        else:
            failed += 1

    # Create one combined dataset for all assets.
    if processed_data:
        combined = pd.concat(
            processed_data,
            ignore_index=True
        )

        combined = combined.sort_values(
            ["exchange", "ticker", "date"]
        ).reset_index(drop=True)

        combined_file = OUTPUT_DIR / "all_assets_processed.csv"
        combined.to_csv(combined_file, index=False)

        print(f"\nCombined dataset saved: {combined_file}")
        print(f"Total processed records: {len(combined)}")

    print("\n========== PROCESSING SUMMARY ==========")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("========================================")
