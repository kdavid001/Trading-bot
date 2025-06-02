from data_fetcher import DataFetcher
from feature_Engine import FeatureEngineer
from news_processor import NewsProcessor
import pandas as pd


def build_dataset(assets):
    """
    Creates unified dataset with market data and news sentiment
    Returns: Dict of DataFrames {asset: processed_data}
    """
    fetcher = DataFetcher()
    engineer = FeatureEngineer()
    processor = NewsProcessor()
    datasets = {}

    for asset in assets:
        symbol = asset['symbol']
        print(f"\n🔍 Processing {symbol}...")

        try:
            # 1. Fetch market data
            market_data = fetcher.get_market_data(symbol)
            if market_data.empty:
                print(f"❌ Empty market data for {symbol} - skipping")
                continue

            # 2. Validate OHLC columns
            if not {'open', 'high', 'low', 'close'}.issubset(market_data.columns):
                print(f"❌ Missing OHLC columns for {symbol}")
                continue

            # 3. Fetch and process news
            news = fetcher.get_news(asset['news_query'])
            sentiment = processor.process_news_batch(news)

            # 4. Feature engineering
            processed_data = engineer.add_technical_features(market_data)
            if processed_data.empty:
                print(f"❌ Feature engineering returned empty DataFrame for {symbol}")
                continue

            # 5. Create targets
            processed_data = engineer.create_targets(processed_data)
            if processed_data.empty:
                print(f"❌ Target creation returned empty DataFrame for {symbol}")
                continue

            # 6. Add sentiment
            processed_data['news_sentiment'] = sentiment
            datasets[symbol] = processed_data

            print(f"✅ Successfully processed {symbol} | Shape: {processed_data.shape}")

        except Exception as e:
            print(f"🔥 Error processing {symbol}: {str(e)}")
            continue

    if not datasets:
        raise ValueError("❌ No valid datasets were created - check previous error messages")

    print("\n=== Dataset Summary ===")
    for symbol, df in datasets.items():
        print(f"{symbol}: {len(df)} rows | Columns: {list(df.columns)}")

    return datasets