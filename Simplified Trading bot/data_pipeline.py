from data_fetcher import DataFetcher
from feature_Engine import FeatureEngineer
from news_processor import NewsProcessor
import pandas as pd
from typing import List, Dict


def build_dataset(assets: List[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
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
    processor = NewsProcessor()
    datasets = {}



    for asset in assets:
        symbol = asset['symbol']
        print(f"\n🔍 Processing {symbol}...")

        try:
            # 1. Fetch and validate market data
            market_data = fetcher.get_market_data(symbol)
            processed_data = engineer.add_technical_features(market_data)
            if market_data.empty:
                print(f"❌ Empty market data for {symbol} - skipping")
                continue

            if not {'open', 'high', 'low', 'close'}.issubset(market_data.columns):
                print(f"❌ Missing OHLC columns for {symbol}")
                continue

            # Check for datetime index
            if not isinstance(market_data.index, pd.DatetimeIndex):
                print(f"ℹ️ Converting index to datetime for {symbol}")
                market_data.index = pd.to_datetime(market_data.index)


            # 2. Fetch and process news (time-aligned)
            news = fetcher.get_news(asset['news_query'])
            if not news.empty:  # Now safe to check .empty
                # Resample to align with market data frequency
                sentiment = news['sentiment'].resample('D').mean().ffill()

                # Merge with market data
                market_data = market_data.merge(
                    sentiment,
                    left_index=True,
                    right_index=True,
                    how='left',
                    suffixes=('', '_news')
                )
                market_data['news_sentiment'] = market_data['sentiment'].ffill().fillna(0)
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

    # # Dataset quality report
    # print("\n=== Dataset Quality Report ===")
    # for symbol, df in datasets.items():
    #     print(f"\n📊 {symbol}:")
    #     print(f"Time range: {df.index.min()} to {df.index.max()}")
    #     print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    #     print(f"Missing values: {df.isna().sum().sum()}")
    #     if 'news_sentiment' in df.columns:
    #         print(f"Sentiment range: {df['news_sentiment'].min():.2f} to {df['news_sentiment'].max():.2f}")

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


# Add this new function to validate the combined dataset
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


# assets = [
#     {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
#     {'symbol': 'ETH/USD', 'news_query': 'Ethereum'},
#     # {'symbol': 'SPY', 'news_query': 'S&P 500'}
# ]
# data = build_dataset(assets)
# # print(f"This is from the build_dataset function : \n {data}")
# combined = []
# for symbol, df in data.items():
#     df = df.copy()
#     df['symbol'] = symbol
#     combined.append(df)
#
# full_df = pd.concat(combined)
# full_df.to_csv("data/combined_data_pipeline.csv", index=True)
# print("✅ Saved all data to combined_data_pipeline.csv")
