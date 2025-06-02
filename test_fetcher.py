# test_fetcher.py
from data_fetcher import DataFetcher

fetcher = DataFetcher()
for symbol in ['BTC/USD', 'EUR/USD', 'SPY']:
    data = fetcher.get_market_data(symbol)
    print(f"\n{symbol}:")
    print("Shape:", data.shape)
    print("First row:\n", data.head(1) if not data.empty else "EMPTY")