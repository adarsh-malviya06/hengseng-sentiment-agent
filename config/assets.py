ASSETS = {
    "HKEX": [
        {"ticker": "0700.HK", "name": "Tencent"},
        {"ticker": "9988.HK", "name": "Alibaba"},
        {"ticker": "3690.HK", "name": "Meituan"},
        {"ticker": "1810.HK", "name": "Xiaomi"},
        {"ticker": "0005.HK", "name": "HSBC"},
    ],
    "KRX": [
        {"ticker": "005930.KS", "name": "Samsung Electronics"},
        {"ticker": "000660.KS", "name": "SK Hynix"},
        {"ticker": "005380.KS", "name": "Hyundai Motor"},
        {"ticker": "066570.KS", "name": "LG Electronics"},
        {"ticker": "035420.KS", "name": "NAVER"},
    ],
}


def get_all_assets():
    """Return all configured stocks with their exchange information."""
    all_assets = []

    for exchange, stocks in ASSETS.items():
        for stock in stocks:
            all_assets.append({**stock, "exchange": exchange})

    return all_assets
