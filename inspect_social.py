
import json
from pathlib import Path

DATA_DIR = Path("data/raw/social")

for filename in ["lihkg_test.json", "naver_test.json"]:

    file_path = DATA_DIR / filename

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")
    print("=" * 60)

    with open(file_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    print("Total records:", len(records))

    if records:
        print("\nAvailable fields:")
        print(list(records[0].keys()))

        print("\nFirst record:")
        print(json.dumps(
            records[0],
            indent=2,
            ensure_ascii=False
        )[:3500])
    else:
        print("No records found.")
