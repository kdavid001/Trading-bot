from data_pipeline import build_dataset
from model_builder_ST import build_direction_model, prepare_dataset, train_model
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from typing import Dict, Tuple

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set mixed precision policy for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def combine_and_validate_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple asset datasets with validation

    Args:
        datasets: Dictionary of {symbol: DataFrame}

    Returns:
        Combined DataFrame with symbol column

    Raises:
        ValueError: If datasets have incompatible structures
    """
    required_columns = {'open', 'high', 'low', 'close'}
    combined = []

    for symbol, df in datasets.items():
        # Validate each dataset
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Dataset for {symbol} is not a DataFrame")

        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} for {symbol}")

        # Add symbol identifier
        df = df.copy()
        df['symbol'] = symbol
        combined.append(df)

    # Combine and validate
    full_df = pd.concat(combined)

    # Check for datetime index
    if not isinstance(full_df.index, pd.DatetimeIndex):
        raise ValueError("Combined DataFrame must have DatetimeIndex")

    # Sort by time and symbol
    full_df = full_df.sort_values(by=['symbol', full_df.index.name or 'timestamp'])

    return full_df


def main():
    try:
        # Configuration
        # assets = [
        #     {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
        #     {'symbol': 'EUR/USD', 'news_query': 'EUR USD'},
        # ]
        assets = [
            {'symbol': 'BTC-USD', 'news_query': 'Bitcoin'},
            {'symbol': 'EURUSD=X', 'news_query': 'EUR USD'},
        ]
        window_size = 60
        epochs = 150
        batch_size = 128

        # Step 1: Build dataset
        print("🛠️ Building dataset...")
        dataset = build_dataset(assets)
        print(f"✅ Dataset built for {len(dataset)} assets")

        # Step 2: Prepare datasets
        print("🧹 Preparing training and validation datasets...")
        full_df = combine_and_validate_datasets(dataset)

        # Verify we have enough data
        min_samples = window_size * 10  # Minimum reasonable training set
        if len(full_df) < min_samples:
            raise ValueError(f"Insufficient data ({len(full_df)} samples). Need at least {min_samples}")

        train_data, val_data = prepare_dataset(full_df, window_size=window_size)

        # Step 3: Build model
        print("🏗️ Building model...")
        input_shape = (window_size, train_data[0].shape[2])
        model = build_direction_model(input_shape)
        model.summary()

        # Step 4: Train model
        print("🚂 Training model...")
        trained_model, history = train_model(
            model,
            train_data,
            val_data,
            epochs=epochs,
            batch_size=batch_size
        )

        print("🎉 Model training complete!")

    except Exception as e:
        print(f"🔥 Critical error in pipeline: {str(e)}")
        raise


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