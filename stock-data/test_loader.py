from stock_loader import stock_data_loader

Loader = stock_data_loader("NVDA" , start_date="2023-01-01")
Loader.download_data()
Loader.Clean_data()
Loader.add_features()
print(Loader.data.columns.tolist())
print(Loader.data.head())

# Test the three new methods
print("=== Latest 5 days ===")
print(Loader.get_latest_n_days(5))

Loader.save_to_csv("nvda_features.csv")
Loader.save_to_sqlite("stocks.db")

# Quick verify CSV and SQLite were created
import os
print("\n=== Files created ===")
print("nvda_features.csv exists:", os.path.exists("nvda_features.csv"))
print("stocks.db exists:", os.path.exists("stocks.db"))