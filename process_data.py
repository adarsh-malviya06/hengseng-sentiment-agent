from processing.price_features import process_all_stock_files


def main():
    print("=" * 55)
    print("HENGSENG SENTIMENT PRICE PREDICTION AGENT")
    print("Milestone 2: Price Data Processing")
    print("=" * 55)

    process_all_stock_files()

    print("\nMilestone 2 execution completed.")


if __name__ == "__main__":
    main()
