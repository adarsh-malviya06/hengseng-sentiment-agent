
import os

from dotenv import load_dotenv
from apify_client import ApifyClient
from exa_py import Exa


load_dotenv()


def check_apify_connection():
    print("\nTesting Apify connection...")

    token = os.getenv("APIFY_API_TOKEN")

    if not token:
        print("[FAILED] APIFY_API_TOKEN is missing in .env")
        return False

    try:
        client = ApifyClient(token)
        account = client.user().get()

        if not account:
            print("[FAILED] Could not retrieve Apify account.")
            return False

        print("[SUCCESS] Apify API connected!")
        return True

    except Exception as error:
        print(f"[FAILED] Apify connection: {error}")
        return False


def check_exa_connection():
    print("\nTesting Exa connection...")

    api_key = os.getenv("EXA_API_KEY")

    if not api_key:
        print("[FAILED] EXA_API_KEY is missing in .env")
        return False

    try:
        exa = Exa(api_key=api_key)

        response = exa.search(
            "latest Hong Kong stock market news",
            num_results=1
        )

        print("[SUCCESS] Exa API connected!")
        print(f"Search results received: {len(response.results)}")
        return True

    except Exception as error:
        print(f"[FAILED] Exa connection: {error}")
        return False


def main():
    print("=" * 50)
    print("HENGSENG AGENT - API CONNECTION TEST")
    print("=" * 50)

    apify_status = check_apify_connection()
    exa_status = check_exa_connection()

    print("\n========== CONNECTION SUMMARY ==========")
    print(f"Apify: {'CONNECTED' if apify_status else 'FAILED'}")
    print(f"Exa:   {'CONNECTED' if exa_status else 'FAILED'}")
    print("========================================")


if __name__ == "__main__":
    main()
