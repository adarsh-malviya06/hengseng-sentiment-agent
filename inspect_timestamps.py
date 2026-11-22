import json
from pathlib import Path

files = {
    "LIHKG": ("data/raw/social/lihkg_test.json", "created_at"),
    "Naver": ("data/raw/social/naver_test.json", "writtenAt"),
    "Telegram 1": ("data/raw/social/telegram_krx_1.json", "date"),
    "Telegram 2": ("data/raw/social/telegram_krx_2.json", "date"),
}

for source, (path, field) in files.items():
    print("\n" + "=" * 55)
    print(source)
    print("=" * 55)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("data", data.get("results", []))

    populated = [r.get(field) for r in data if r.get(field)]

    print("Total records:", len(data))
    print("Records with timestamp:", len(populated))
    print("Records without timestamp:", len(data) - len(populated))
    print("First 5 timestamps:", populated[:5])
