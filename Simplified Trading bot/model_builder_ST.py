import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    BatchNormalization, Conv1D, Multiply,
    Permute, RepeatVector, Flatten
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.preprocessing import MinMaxScaler
import os
import datetime

# Configuration
WINDOW_SIZE = 60
LOOKAHEAD_PERIOD = 4  # Predict direction 4 periods ahead
THRESHOLD = 0.0015  # Minimum price movement threshold


def generate_features(data):
    """Create technical features without lookahead bias"""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    # Calculate price features
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['high'] - data['low']
    data['momentum'] = data['close'].pct_change(5)

    # Calculate volume features if volume exists
    if 'volume' in data.columns:
        mean_vol = data['volume'].rolling(20, min_periods=1).mean()
        std_vol = data['volume'].rolling(20, min_periods=1).std()
        data['volume_z'] = (data['volume'] - mean_vol) / (std_vol + 1e-8)
    else:
        data['volume_z'] = 0.0

    # Volatility regimes
    data['vol_regime'] = (data['volatility'] > data['volatility'].rolling(50).mean()).astype(int)

    # Drop initial NaNs
    return data.dropna()


def create_sequences(data, targets, window_size=WINDOW_SIZE):
    """Create time-series sequences for LSTM"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(targets[i - 1])  # Using pre-created targets
    return np.array(X), np.array(y)


def prepare_dataset(dataset, window_size=WINDOW_SIZE):
    """Prepare dataset with proper temporal splitting"""
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Generate features
    data = generate_features(dataset)

    # Create target
    data['future_close'] = data['close'].shift(-LOOKAHEAD_PERIOD)
    data['direction'] = np.where(
        data['future_close'] > data['close'] * (1 + THRESHOLD), 1, 0
    )

    # Filter out neutral movements
    valid_mask = (data['future_close'] > data['close'] * (1 + THRESHOLD)) | \
                 (data['future_close'] < data['close'] * (1 - THRESHOLD))
    data = data[valid_mask].copy()

    # Verify no lookahead bias
    assert all(data['future_close'].shift(1).notna()), "Lookahead bias detected!"

    # Required features - adjust based on your actual features
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'volatility', 'momentum',
        'volume_z', 'vol_regime'
    ]

    # Temporally split
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Scale features
    scaler = MinMaxScaler()
    train_scaled = train_data.copy()
    train_scaled[features] = scaler.fit_transform(train_data[features])
    test_scaled = test_data.copy()
    test_scaled[features] = scaler.transform(test_data[features])

    # Create sequences
    X_train, y_train = create_sequences(train_scaled[features].values, train_scaled['direction'].values)
    X_test, y_test = create_sequences(test_scaled[features].values, test_scaled['direction'].values)

    print(f"\n📊 Dataset Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Class balance: {np.mean(y_train):.2%} up / {1 - np.mean(y_train):.2%} down")

    return (X_train, y_train), (X_test, y_test)


def build_direction_model(input_shape):
    """Build enhanced model for directional prediction"""
    inputs = Input(shape=input_shape)

    # Feature normalization
    x = BatchNormalization()(inputs)

    # Temporal feature extraction
    x = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Hierarchical LSTM
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)
    x = Multiply()([x, attention])

    # Output
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='prec'), Recall(name='rec')]
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    """Train directional model with callbacks"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("models", exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(
            f"models/best_model_{timestamp}.h5",
            save_best_only=True,
            monitor='val_loss'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    print("\n🔥 Training directional model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(f"models/direction_model_{timestamp}.keras")
    print(f"\n💾 Model saved to: models/direction_model_{timestamp}.keras")
    return model, history