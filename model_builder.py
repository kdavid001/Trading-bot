import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from attention_layer import Attention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os
import datetime


def create_sequences(data, targets, window_size=60):
    """Create time-series sequences for LSTM"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def build_model(input_shape, num_classes=3):
    """
    Build hybrid CNN-LSTM model with attention
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # 1D Convolution for local pattern extraction
    conv1 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3,
        activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)

    # LSTM layers with attention
    lstm1 = LSTM(128, return_sequences=True)(conv1)
    attention = Attention(return_sequences=True)(lstm1)
    lstm2 = LSTM(64, return_sequences=False)(attention)

    # Multi-task output heads
    volatility = Dense(1, activation='sigmoid', name='volatility')(lstm2)
    trend = Dense(num_classes, activation='softmax', name='trend')(lstm2)
    reversal = Dense(1, activation='sigmoid', name='reversal')(lstm2)

    # Create model
    model = Model(inputs=inputs, outputs=[volatility, trend, reversal])

    # Custom loss weights
    loss_weights = {'volatility': 0.4, 'trend': 0.4, 'reversal': 0.2}

    # Custom metrics
    metrics = {
        'trend': ['accuracy', Precision(name='precision'), Recall(name='recall')],
        'reversal': ['accuracy', Precision(name='precision'), Recall(name='recall')]
    }

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'volatility': 'mse',
            'trend': 'categorical_crossentropy',
            'reversal': 'binary_crossentropy'
        },
        loss_weights=loss_weights,
        metrics=metrics
    )

    return model


def prepare_datasets(dataset, window_size=60, test_size=0.2):
    """Prepare datasets with guaranteed feature consistency"""
    # 1. Define the complete feature set we expect
    mandatory_features = [
        'open', 'high', 'low', 'close', 'returns', 'volatility',
        'rsi', 'macd', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
        'kst', 'squeeze', 'news_sentiment', 'volume'  # Now includes volume
    ]

    # 2. Initialize containers
    all_X, all_y_vol, all_y_trend, all_y_reversal = [], [], [], []

    # 3. Process each asset
    for symbol, data in dataset.items():
        print(f"\nProcessing {symbol}...")

        # Skip empty datasets
        if data.empty:
            print(f"⚠️ Empty data for {symbol} - skipping")
            continue

        # Ensure all features exist (fill missing with 0)
        processed_features = data.copy()
        for feature in mandatory_features:
            if feature not in processed_features.columns:
                print(f"⚠️ Adding missing feature: {feature}")
                processed_features[feature] = 0

        # Verify targets exist
        required_targets = ['vol_target', 'trend_target', 'reversal_target']
        if not all(t in processed_features.columns for t in required_targets):
            print(f"⚠️ Missing targets in {symbol} - skipping")
            continue

        try:
            # Separate features and targets
            features = processed_features[mandatory_features]
            targets = processed_features[required_targets]

            # Scale features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)

            # Create sequences
            X, yv = create_sequences(scaled_features, targets['vol_target'].values, window_size)
            _, yt = create_sequences(scaled_features, targets['trend_target'].values, window_size)
            _, yr = create_sequences(scaled_features, targets['reversal_target'].values, window_size)

            # One-hot encode trend
            encoder = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False)
            yt = encoder.fit_transform(yt.reshape(-1, 1))

            # Store results
            all_X.append(X)
            all_y_vol.append(yv)
            all_y_trend.append(yt)
            all_y_reversal.append(yr)

            print(f"✅ {symbol} processed | Features: {X.shape[2]} | Samples: {X.shape[0]}")

        except Exception as e:
            print(f"🔥 Processing failed for {symbol}: {str(e)}")
            continue

    # 4. Validate we have data
    if not all_X:
        raise ValueError("❌ No valid datasets created")

    # 5. Combine all assets
    print("\nFinal feature dimensions:")
    for i, X in enumerate(all_X):
        print(f"{list(dataset.keys())[i]}: {X.shape[2]} features")

        # Combine all assets
        X_combined = np.concatenate(all_X)
        y_vol_combined = np.concatenate(all_y_vol)
        y_trend_combined = np.concatenate(all_y_trend)
        y_reversal_combined = np.concatenate(all_y_reversal)

    splits = train_test_split(
        X_combined,
        y_vol_combined,
        y_trend_combined,
        y_reversal_combined,
        test_size=test_size,
        shuffle=False
    )

    # Unpack all splits
    X_train, X_val, y_vol_train, y_vol_val, y_trend_train, y_trend_val, y_reversal_train, y_reversal_val = splits

    # Return in expected format
    return (X_train, [y_vol_train, y_trend_train, y_reversal_train]), \
        (X_val, [y_vol_val, y_trend_val, y_reversal_val])


def train_model(model, train_data, val_data, epochs=100, batch_size=64):
    """Train model with callbacks"""
    # Create log directory with timestamp
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs("models", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join("models", "best_model.h5"),
            save_best_only=True,
            monitor='val_loss'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train model
    history = model.fit(
        x=train_data[0],
        y={
            'volatility': train_data[1][0],
            'trend': train_data[1][1],
            'reversal': train_data[1][2]
        },
        validation_data=(val_data[0], {
            'volatility': val_data[1][0],
            'trend': val_data[1][1],
            'reversal': val_data[1][2]
        }),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Save final model
    model.save(os.path.join("models", "final_model.h5"))

    return model, history
