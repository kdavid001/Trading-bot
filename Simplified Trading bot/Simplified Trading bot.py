import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_pipeline import build_dataset, combine_datasets, validate_combined_data
from model_builder_ST import build_direction_model, prepare_dataset, train_model, generate_features
from news_scraper import News_scraper

# from News_analysis import AssetNewsFetcher
# news_fetcher = AssetNewsFetcher()

news_fetcher = News_scraper()

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.mixed_precision.set_global_policy('float32')

# User data
trading_type = "forex"
if trading_type == "forex":
    # TODO: move to user data section
    assets = [
        {'symbol': 'EURUSD=X', 'news_query': 'Euro Dollar'},
        # {'symbol': 'USDJPY=X', 'news_query': 'Dollar Yen'},
        # {'symbol': 'GBPUSD=X', 'news_query': 'Pound Dollar'},
        # {'symbol': 'USDCHF=X', 'news_query': 'Dollar Swiss Franc'},
        # {'symbol': 'AUDUSD=X', 'news_query': 'Aussie Dollar'},
        # {'symbol': 'USDCAD=X', 'news_query': 'Dollar Canadian'}
    ]
elif trading_type == "crypto":
    # TODO: move to user data section
    assets = [
        {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
        # {'symbol': 'ETH/USD', 'news_query': 'Ethereum'},
        # {'symbol': 'SPY', 'news_query': 'S&P 500'}
    ]


def main():
    try:
        # TODO: change this to collect input from user later on then store the input in a list<dict>

        epochs = 100
        batch_size = 128
        min_samples = 1000  # Minimum samples per asset
        lookback_years = 10
        LOOKAHEAD_PERIOD = 1  # Predict direction 4 periods ahead (4 days ahead)
        window_size = 120
        # THRESHOLD = 0.0015  # Minimum price movement threshold
        THRESHOLD = 0.0015


        # Step 1: Build dataset
        print("🛠️ Building dataset...")
        raw_datasets = build_dataset(assets, lookback_years=lookback_years, trading_type=trading_type)
        print(f"✅ Dataset built for {len(raw_datasets)} assets")

        # Step 2: Combine and validate
        print("🧹 Combining and validating datasets...")
        full_df = combine_datasets(raw_datasets)
        full_df = validate_combined_data(full_df, trading_type, min_samples)
        full_df.to_csv("data/combined_data_pipeline.csv", index=True)
        print("✅ Saved all data to combined_data_pipeline.csv")
        # Step 3: Prepare for training
        print("⚙️ Preparing training data...")
        # print(full_df.describe())
        # print(full_df.columns)
        # print(full_df.index.dtype)
        # return full_df.head(100)
        (X_train, y_train), (X_val, y_val) = prepare_dataset(
            full_df, trading_type=trading_type,
            LOOKAHEAD_PERIOD=LOOKAHEAD_PERIOD,
            THRESHOLD=THRESHOLD,
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
            print("- Features after engineering:", generate_features(full_df, trading_type).shape)
            print("- Unique symbols:", full_df['symbol'].unique())
            print("- Date range:", full_df.index.min(), "to", full_df.index.max())
            raise ValueError("Empty training data - see debug output above")
        print("🚂 Training model...")
        from sklearn.utils import class_weight
        y_train_adj = y_train + 1  # -1 → 0, 0 → 1, 1 → 2
        y_val_adj = y_val + 1
        classes = np.unique(y_train_adj)
        cw = class_weight.compute_class_weight('balanced', classes=classes, y=y_train_adj)
        class_weights = dict(zip(classes, cw))
        # Shift labels so that -1 → 0, 0 → 1, 1 → 2
        print("Unique labels:", np.unique(y_train_adj))
        print("Label dtype:", y_train_adj.dtype)
        print("Any NaNs in y?", np.isnan(y_train_adj).any())
        print("Any NaNs in X?", np.isnan(X_train).any())

        y_train_adj = y_train_adj.astype('int64')
        y_val_adj = y_val_adj.astype('int64')
        trained_model, history = train_model(
            model,
            X_train, y_train_adj,
            X_val, y_val_adj,
            trading_type,
            class_weights,
            asset_name=str(assets[0]['symbol']),
            epochs=epochs,
            batch_size=batch_size
        )
        print("🎉 Training complete!")

        y_pred = trained_model.predict(X_val)
        # y_pred_labels = np.argmax(y_pred, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1) - 1
        print(confusion_matrix(y_val, y_pred_labels))
        print("Accuracy:", accuracy_score(y_val, y_pred_labels))
        print(classification_report(y_val, y_pred_labels))
        print(f"to check imbalance{np.bincount(y_train)}")
        print(f"to check imbalance{np.bincount(y_val)}")
        # TODO: Stopping the model here until I can see an improvement
        return "Done"

        asset_name = assets[0]['news_query']
        asset_symbol = assets[0]['symbol']
        print(f"Fetching news for {asset_name}")
        print("0 done -> next step")

        # for crypto same news affects the general market, but it is different for forex
        news = news_fetcher.get_latest_article(asset_symbol, trading_type)
        print("1 done -> next step")
        print(news)
        sentiment_result = news_fetcher.process_sentiment(news, trading_type)
        print(sentiment_result)
        print("2 done -> result printed, step")

        # Make prediction with sentiment
        prediction = make_prediction(window_size, trained_model, news, sentiment_result)
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


def make_prediction(window_size, model, market_data: pd.DataFrame, sentiment_score):
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
    features = generate_features(market_data, trading_type)

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
