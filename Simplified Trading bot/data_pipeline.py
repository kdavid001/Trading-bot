from data_fetcher import DataFetcher
from feature_Engine import FeatureEngineer
from news_processor import NewsProcessor
import pandas as pd
from typing import List, Dict


def build_dataset(assets: List[Dict[str, str]], lookback_years: int, trading_type: str):
    """
    Creates unified dataset with market data and news sentiment

    Args:
        assets: List of dictionaries with 'symbol' and 'news_query' keys

    Returns:
        Dictionary of processed DataFrames {asset_symbol: processed_data}

    Raises:
        ValueError: If no valid datasets are created
    """
    fetcher = DataFetcher()
    engineer = FeatureEngineer()
    news_processor = NewsProcessor()
    datasets = {}



    for asset in assets:
        symbol = asset['symbol']
        print(f"\n🔍 Processing {symbol}...")

        try:
            # 1. Fetch and validate market data
            market_data = fetcher.get_market_data(symbol, lookback_years=lookback_years, trading_type=trading_type)
            processed_data = engineer.add_technical_features(market_data)
            if market_data.empty:
                print(f"❌ Empty market data for {symbol} - skipping")
                continue

            # if not {'open', 'high', 'low', 'close', 'volume'}.issubset(market_data.columns):
            #     print(f"❌ Missing OHLCv columns for {symbol}")
            #     continue

            # Check for datetime index
            if not isinstance(market_data.index, pd.DatetimeIndex):
                print(f"ℹ️ Converting index to datetime for {symbol}")
                market_data.index = pd.to_datetime(market_data.index)

            # 2. Fetch and process news (time-aligned)
            news = fetcher.get_news(asset['news_query'])

            if not news.empty:
                print(f"📰 Processing {len(news)} news articles for {symbol}")

                # 1. Check for required columns and create fallbacks if missing
                if 'description' not in news.columns:
                    print("⚠️ 'description' column missing - using empty strings")
                    news['description'] = ""

                # if 'title' not in news.columns:
                #     print("⚠️ 'title' column missing - cannot process sentiment")
                #     market_data['news_sentiment'] = 0.0
                #     print("Taken to 0")
                #     return market_data

                # 2. Prepare articles safely
                articles_to_process = []
                for _, row in news.iterrows():
                    articles_to_process.append({
                        'title': str(row.get('title', '')),
                        'description': str(row.get('description', ''))
                    })

                # 3. Process in batches with error handling
                batch_size = 100
                sentiments = []

                for i in range(0, len(articles_to_process), batch_size):
                    batch = articles_to_process[i:i + batch_size]
                    try:
                        batch_sentiments = news_processor.process_news_batch(batch)
                        sentiments.extend(batch_sentiments)
                    except Exception as e:
                        print(f"⚠️ Error processing batch {i // batch_size + 1}: {str(e)}")
                        sentiments.extend([0.0] * len(batch))  # Neutral fallback

                # 4. Add sentiments to DataFrame
                news['sentiment'] = sentiments

                # 5. Process sentiment timeline
                news.index = pd.to_datetime(news.index)
                daily_sentiment = news['sentiment'].resample('D').mean().ffill()

                # 6. Merge with market data
                market_data = market_data.merge(
                    daily_sentiment.rename('news_sentiment'),
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                market_data['news_sentiment'] = market_data['news_sentiment'].fillna(0)

                print(f"✅ Added news sentiment ({daily_sentiment.notna().sum()} days with sentiment)")
            else:
                print(f"⚠️ No news data for {symbol}")
                market_data['news_sentiment'] = 0.0
            # 3. Feature engineering
            processed_data = engineer.add_technical_features(market_data)
            if processed_data.empty:
                print(f"❌ Feature engineering returned empty DataFrame for {symbol}")
                continue

            # 4. Create targets (uncomment when ready)
            create_targets = False
            if create_targets:
                processed_data = engineer.create_targets(processed_data)
            if processed_data.empty:
                print(f"❌ Target creation returned empty DataFrame for {symbol}")
                continue

            # Add symbol identifier
            processed_data['symbol'] = symbol

            datasets[symbol] = processed_data
            print(f"✅ Successfully processed {symbol} | Shape: {processed_data.shape}")

        except Exception as e:
            print(f"🔥 Error processing {symbol}: {str(e)}")
            continue

    if not datasets:
        raise ValueError("❌ No valid datasets were created - check previous error messages")

    # Dataset quality report
    print("\n=== Dataset Quality Report ===")
    for symbol, df in datasets.items():
        print(f"\n📊 {symbol}:")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        print(f"Missing values: {df.isna().sum().sum()}")
        if 'news_sentiment' in df.columns:
            print(f"Sentiment range: {df['news_sentiment'].min():.2f} to {df['news_sentiment'].max():.2f}")

    return datasets


def combine_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple asset datasets into one unified DataFrame

    Args:
        datasets: Dictionary of DataFrames from build_dataset()

    Returns:
        Combined DataFrame with symbol column preserved
    """
    combined = []
    for symbol, df in datasets.items():
        df = df.copy()
        df['symbol'] = symbol  # Ensure symbol column exists
        combined.append(df)

    full_df = pd.concat(combined)

    # Sort by timestamp and symbol
    full_df = full_df.sort_values(by=['symbol', full_df.index.name or 'timestamp'])
    # Check for duplicates
    duplicates = full_df.duplicated(subset=['symbol', full_df.index.name or 'timestamp'])
    if duplicates.any():
        print(f"⚠️ Found {duplicates.sum()} duplicate timestamps - keeping first occurrence")
        full_df = full_df[~duplicates]

    return full_df


def validate_combined_data(full_df: pd.DataFrame, min_samples_per_asset: int = 1000) -> pd.DataFrame:
    """Validate the combined dataset meets minimum requirements"""
    if not isinstance(full_df.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")

    # Check each symbol has enough data
    symbol_counts = full_df['symbol'].value_counts()
    for symbol, count in symbol_counts.items():
        if count < min_samples_per_asset:
            raise ValueError(f"Symbol {symbol} only has {count} samples (min {min_samples_per_asset})")

    # Check required columns
    required_cols = {'open', 'high', 'low', 'close', 'symbol'}
    missing = required_cols - set(full_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return full_df.sort_index()

if __name__ == "__main__":
    trading_type = 'forex'
    # assets = [
        # {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
    #     {'symbol': 'ETH/USD', 'news_query': 'Ethereum'},
    #     # {'symbol': 'SPY', 'news_query': 'S&P 500'}
    # ]
    #
    assets = [
                {'symbol': 'EURUSD=X', 'news_query': 'Euro Dollar'},
                # {'symbol': 'USDJPY=X', 'news_query': 'Dollar Yen'},
                # {'symbol': 'GBPUSD=X', 'news_query': 'Pound Dollar'},
                # {'symbol': 'USDCHF=X', 'news_query': 'Dollar Swiss Franc'},
                # {'symbol': 'AUDUSD=X', 'news_query': 'Aussie Dollar'},
                # {'symbol': 'USDCAD=X', 'news_query': 'Dollar Canadian'}
            ]

    data = build_dataset(assets, lookback_years=1, trading_type=trading_type)
    # print(f"This is from the build_dataset function : \n {data}")
    combined = []
    for symbol, df in data.items():
        df = df.copy()
        df['symbol'] = symbol
        combined.append(df)

    full_df = pd.concat(combined)
    full_df.to_csv("data/combined_data_pipeline.csv", index=True)
    print("✅ Saved all data to combined_data_pipeline.csv")
