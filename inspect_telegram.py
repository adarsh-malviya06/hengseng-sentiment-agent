
import json
from pathlib import Path

files = [
    "data/raw/social/telegram_krx_1.json",
    "data/raw/social/telegram_krx_2.json",
]

for file_path in files:
    print("\n" + "=" * 70)
    print(f"FILE: {file_path}")
    print("=" * 70)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("data", data.get("results", []))
    else:
        records = []

    print("Total records:", len(records))

    if records:
        print("\nFirst record:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))

        print("\nAvailable fields:")
        print(list(records[0].keys()))
