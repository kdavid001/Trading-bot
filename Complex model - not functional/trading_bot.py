from data_pipeline import build_dataset
from model_builder import build_model, prepare_datasets, train_model
import tensorflow as tf
import os
import numpy as np

# Set mixed precision policy for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def main():
    # Define assets to track
    assets = [
        {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
        {'symbol': 'EUR/USD', 'news_query': 'EUR USD'},
        # {'symbol': 'SPY', 'news_query': 'S&P 500'}
    ]

    # Step 1: Build dataset
    print("Building dataset...")
    dataset = build_dataset(assets)
    print(f"Dataset built for {len(dataset)} assets")

    # Step 2: Prepare datasets
    print("Preparing training and validation datasets...")
    train_data, val_data = prepare_datasets(dataset, window_size=60)

    # Step 3: Build model
    print("Building model...")
    input_shape = train_data[0].shape[1:]  # Get shape from training data
    model = build_model(input_shape)
    model.summary()

    # Step 4: Train model
    print("Training model...")
    trained_model, history = train_model(
        model,
        train_data,
        val_data,
        epochs=150,
        batch_size=128
    )

    print("Model training complete!")


if __name__ == "__main__":
    # Set GPU growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()