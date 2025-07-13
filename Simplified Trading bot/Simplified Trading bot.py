import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from data_pipeline import build_dataset
from model_builder_ST import build_direction_model, prepare_dataset, train_model, generate_features
# from News_analysis import AssetNewsFetcher
# news_fetcher = AssetNewsFetcher()

from news_scraper import News_scraper

news_fetcher = News_scraper()

# Set random seeds for reproducibility
SEED = 42
window_size = 60
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
trading_type = "forex"
tf.keras.mixed_precision.set_global_policy('mixed_float16')


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
    if trading_type == "forex":
        required_cols = {'open', 'high', 'low', 'close', 'symbol'}
        missing = required_cols - set(full_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    elif trading_type == "crypto":
        required_cols = {'open', 'high', 'low', 'symbol', 'volume'}
        missing = required_cols - set(full_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    return full_df.sort_index()


def combine_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple asset datasets into one unified DataFrame
    with proper datetime handling
    """
    combined = []
    for symbol, df in datasets.items():
        df = df.copy()

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                df.index = pd.to_datetime(df.index)

        df['symbol'] = symbol
        combined.append(df)

    full_df = pd.concat(combined)

    # Sort by datetime and symbol
    full_df = full_df.sort_values(by=['symbol', full_df.index.name or 'datetime'])

    # Check for duplicates
    duplicates = full_df.reset_index().duplicated(subset=['symbol', full_df.index.name or 'index'])
    if duplicates.any():
        print(f"⚠️ Found {duplicates.sum()} duplicate timestamps - keeping first occurrence")
        full_df = full_df[~duplicates]

    return full_df


def main():
    global assets
    try:
        # TODO: change this to collect input from user later on then store the input in a list<dict>
        if trading_type == "forex":
            assets = [
                # {'symbol': 'EURUSD=X', 'news_query': 'Euro Dollar'},
                {'symbol': 'USDJPY=X', 'news_query': 'Dollar Yen'},
                # {'symbol': 'GBPUSD=X', 'news_query': 'Pound Dollar'},
                # {'symbol': 'USDCHF=X', 'news_query': 'Dollar Swiss Franc'},
                # {'symbol': 'AUDUSD=X', 'news_query': 'Aussie Dollar'},
                # {'symbol': 'USDCAD=X', 'news_query': 'Dollar Canadian'}
            ]
        elif trading_type == "crypto":
            assets = [
                {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
                # {'symbol': 'ETH/USD', 'news_query': 'Ethereum'},
                # {'symbol': 'SPY', 'news_query': 'S&P 500'}
            ]

        window_size = 60
        epochs = 150
        batch_size = 128
        min_samples = 1000  # Minimum samples per asset
        lookback_years = 10

        # Step 1: Build dataset
        print("🛠️ Building dataset...")
        raw_datasets = build_dataset(assets, lookback_years=lookback_years, trading_type=trading_type)
        print(f"✅ Dataset built for {len(raw_datasets)} assets")
        # In your main() function, after building the dataset:
        # print("\n=== Data Sample ===")
        # for symbol, df in raw_datasets.items():
        #     print(f"\n{symbol} data:")
        #     print("Columns:", df.columns.tolist())
        #     print("Index type:", type(df.index))
        #     print("First 5 rows:")
        #     print(df.head())

        # Step 2: Combine and validate
        print("🧹 Combining and validating datasets...")
        full_df = combine_datasets(raw_datasets)
        full_df = validate_combined_data(full_df, min_samples)
        full_df.to_csv("data/combined_data_pipeline.csv", index=True)
        print("✅ Saved all data to combined_data_pipeline.csv")
        # Step 3: Prepare for training
        print("⚙️ Preparing training data...")
        (X_train, y_train), (X_val, y_val) = prepare_dataset(
            full_df,
            window_size=window_size
        )

        # Step 4: Build model
        print("🏗️ Building model...")
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        if X_train.shape[0] == 0:
            raise ValueError("❌ X_train is empty — check window size, data cleaning, or dataset preparation logic.")
        input_shape = (window_size, X_train.shape[2])
        print(input_shape, len(input_shape))
        model = build_direction_model(input_shape)
        model.summary()

        # Step 5: Train
        # Before training
        if len(X_train) == 0 or len(y_train) == 0:
            print("❌ Empty training data - debugging info:")
            print("- Original data shape:", full_df.shape)
            print("- Features after engineering:", generate_features(full_df).shape)
            print("- Unique symbols:", full_df['symbol'].unique())
            print("- Date range:", full_df.index.min(), "to", full_df.index.max())
            raise ValueError("Empty training data - see debug output above")
        print("🚂 Training model...")
        trained_model, history = train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            trading_type,
            epochs=epochs,
            batch_size=batch_size
        )

        print("🎉 Training complete!")

        asset_name = assets[0]['news_query']
        asset_symbol = assets[0]['symbol']
        print(f"Fetching news for {asset_name}")
        print("0 done -> next step")

        # for crypto same news affects the general market nut it is different for forex
        news = news_fetcher.get_latest_article(asset_symbol, trading_type)
        print("1 done -> next step")
        print(news)
        sentiment_result = news_fetcher.process_sentiment(news, trading_type)
        print(sentiment_result)
        print("2 done -> result printed, step")
        # latest_market_data = raw_datasets[assets[0]['symbol']].iloc[-window_size * 2:]  # Last 2 windows
        # Make prediction with sentiment
        prediction = make_prediction(trained_model, news, sentiment_result)
        print("3 done -> next step")
        print("\n=== Prediction with Sentiment ===")
        print(f"Base Prediction: {prediction['base_prediction']:.2%}")
        print(f"Sentiment Score: {prediction['sentiment_score']:.2f}")
        print(f"Adjusted Prediction: {prediction['adjusted_prediction']:.2%}")
        print(f"Sentiment Weight: {prediction['sentiment_weight']:.0%}")

        # Trading decision logic example
        if prediction['adjusted_prediction'] > 0.6:
            print("✅ Strong buy signal (positive sentiment)")
        elif prediction['adjusted_prediction'] < 0.4:
            print("🚨 Strong sell signal (negative sentiment)")
        else:
            print("➖ Neutral signal")
    except Exception as e:
        print(f"🔥 Pipeline failed: {str(e)}")
        raise


def make_prediction(model, market_data: pd.DataFrame, sentiment_score):
    """
    Make prediction using model and adjust with sentiment score
    Args:
        model: Trained TensorFlow model
        market_data: Latest OHLCV data as DataFrame
        sentiment_score: Float between -1 (negative) and 1 (positive)
    Returns:
        Dictionary with prediction details
    """
    # Generate features from market data
    features = generate_features(market_data)

    # Create input sequence
    sequence = np.array([features.values[-window_size:]]).astype(np.float32)  # Ensure correct dtype45
    # Get base model prediction
    base_prediction = float(model.predict(sequence)[0][0])

    # Adjust prediction with sentiment (20% weight)
    sentiment_weight = 0.2
    adjusted_prediction = base_prediction + (sentiment_score * sentiment_weight)
    adjusted_prediction = np.clip(adjusted_prediction, 0, 1)  # Keep between 0-1

    return {
        'base_prediction': base_prediction,
        'sentiment_score': sentiment_score,
        'adjusted_prediction': adjusted_prediction,
        'sentiment_weight': sentiment_weight
    }


if __name__ == "__main__":
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Optional: Set specific GPU
            # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")

    # Run pipeline
    main()
