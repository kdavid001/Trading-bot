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
# THRESHOLD = 0.0015  # Minimum price movement threshold
THRESHOLD = 0.003  # Increased from 0.0015

def create_sequences(data, targets, window_size=WINDOW_SIZE):
    """Create time-series sequences for LSTM"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(targets[i - 1])  # Using pre-created targets
    return np.array(X), np.array(y)


def generate_features(data):
    """Safe feature engineering with fallbacks for all features"""
    data = data.copy()

    # 1. Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = set(required_cols) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Calculate basic features with shift(1) to prevent lookahead
    data['returns'] = data['close'].pct_change().shift(1)
    data['volatility'] = (data['high'] - data['low']).shift(1)
    data['momentum'] = data['close'].pct_change(5).shift(1)

    # 3. Volume features (with fallback)
    if 'volume' in data.columns:
        vol_mean = data['volume'].rolling(20, min_periods=1).mean().shift(1)
        vol_std = data['volume'].rolling(20, min_periods=1).std().shift(1)
        data['volume_z'] = (data['volume'] - vol_mean) / (vol_std + 1e-8)
    else:
        data['volume_z'] = 0.0

    # 4. Volatility regime (with fallback)
    if 'volatility' in data.columns:
        data['vol_regime'] = (data['volatility'] > data['volatility'].rolling(50).mean().shift(1)).astype(int)
    else:
        data['vol_regime'] = 0

    # 5. Only drop rows where essential features are missing
    essential = ['returns', 'volatility', 'momentum']
    return data.dropna(subset=essential)

# def prepare_dataset(dataset, window_size=WINDOW_SIZE):
#     """More robust dataset preparation with debugging"""
#     print("\n=== Initial Data ===")
#     print(f"Total samples: {len(dataset)}")
#     print(f"Columns: {dataset.columns.tolist()}")
#
#     # Feature engineering
#     data = generate_features(dataset)
#     print("\n=== After Feature Engineering ===")
#     print(f"Samples remaining: {len(data)}")
#
#     # Create target
#     data['future_close'] = data.groupby('symbol')['close'].shift(-LOOKAHEAD_PERIOD)
#     data['direction'] = np.where(
#         data['future_close'] > data['close'] * (1 + THRESHOLD), 1, 0
#     )
#     print("\n=== After Target Creation ===")
#     print(f"Samples with targets: {len(data.dropna(subset=['direction']))}")
#
#     # Temporal split
#     split_time = data.index[int(len(data) * 0.8)]
#     train_data = data[data.index < split_time]
#     val_data = data[data.index >= split_time]
#     print("\n=== After Temporal Split ===")
#     print(f"Training samples: {len(train_data)}")
#     print(f"Validation samples: {len(val_data)}")
#
#     # Scale each symbol separately
#     features = ['open', 'high', 'low', 'close', 'returns', 'volatility',
#                 'momentum', 'volume_z', "vol_regime"]
#
#     scalers = {}
#     scaled_dfs = []
#
#     for symbol, group in train_data.groupby('symbol'):
#         scaler = MinMaxScaler()
#         scaled = group.copy()
#         scaled[features] = scaler.fit_transform(group[features])
#         scalers[symbol] = scaler
#         scaled_dfs.append(scaled)
#
#     train_scaled = pd.concat(scaled_dfs)
#
#     # Scale validation data
#     val_scaled = []
#     for symbol, group in val_data.groupby('symbol'):
#         if symbol in scalers:  # Only use symbols seen in training
#             scaled = group.copy()
#             scaled[features] = scalers[symbol].transform(group[features])
#             val_scaled.append(scaled)
#
#     val_scaled = pd.concat(val_scaled)
#
#     # Create sequences
#     X_train, y_train = create_sequences(train_scaled[features].values, train_scaled['direction'].values)
#     X_val, y_val = create_sequences(val_scaled[features].values, val_scaled['direction'].values)
#
#     print("\n=== Final Shapes ===")
#     print(f"X_train: {X_train.shape}")
#     print(f"y_train: {y_train.shape}")
#     print(f"X_val: {X_val.shape}")
#     print(f"y_val: {y_val.shape}")
#
#     return (X_train, y_train), (X_val, y_val)

def prepare_dataset(dataset, window_size=WINDOW_SIZE):
    """More robust dataset preparation"""
    try:
        # Feature engineering
        data = generate_features(dataset)

        # Create target
        data['future_close'] = data.groupby('symbol')['close'].shift(-LOOKAHEAD_PERIOD)
        data['direction'] = np.where(
            data['future_close'] > data['close'] * (1 + THRESHOLD), 1, 0
        )

        # Temporal split
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]

        # Define features - only use existing ones
        possible_features = ['open', 'high', 'low', 'close', 'returns',
                             'volatility', 'momentum', 'volume_z', 'vol_regime']
        features = [f for f in possible_features if f in data.columns]

        # Scale each symbol separately
        scalers = {}
        train_scaled = []

        for symbol, group in train_data.groupby('symbol'):
            scaler = MinMaxScaler()
            scaled = group.copy()
            scaled[features] = scaler.fit_transform(group[features])
            scalers[symbol] = scaler
            train_scaled.append(scaled)

        train_scaled = pd.concat(train_scaled)

        # Scale validation data
        val_scaled = []
        for symbol, group in val_data.groupby('symbol'):
            if symbol in scalers:
                scaled = group.copy()
                scaled[features] = scalers[symbol].transform(group[features])
                val_scaled.append(scaled)

        val_scaled = pd.concat(val_scaled)

        # Create sequences
        X_train, y_train = create_sequences(train_scaled[features].values, train_scaled['direction'].values)
        X_val, y_val = create_sequences(val_scaled[features].values, val_scaled['direction'].values)

        return (X_train, y_train), (X_val, y_val)

    except Exception as e:
        print(f"Error in prepare_dataset: {str(e)}")
        # Return empty arrays if something fails
        empty = np.array([])
        return (empty, empty), (empty, empty)
#
# def build_direction_model(input_shape):
#     """Build enhanced model for directional prediction"""
#     inputs = Input(shape=input_shape)
#
#     # Feature normalization
#     x = BatchNormalization()(inputs)
#
#     # Temporal feature extraction
#     x = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#
#     # Hierarchical LSTM
#     x = LSTM(128, return_sequences=True)(x)
#     x = Dropout(0.3)(x)
#     x = LSTM(64)(x)
#
#     # Attention mechanism
#     attention = Dense(1, activation='tanh')(x)
#     attention = Flatten()(attention)
#     # attention = tf.keras.layers.Activation('softmax')(attention)
#     attention = tf.keras.layers.Activation('sigmoid')(attention)
#     attention = RepeatVector(64)(attention)
#     attention = Permute([2, 1])(attention)
#     x = Multiply()([x, attention])
#
#     # Output
#     output = Dense(1, activation='sigmoid')(x)
#
#     model = Model(inputs=inputs, outputs=output)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005),
#         loss='binary_crossentropy',
#         metrics=['accuracy', Precision(name='prec'), Recall(name='rec')]
#     )
#     return model
def build_direction_model(input_shape):
    """Build enhanced model for directional prediction"""
    inputs = Input(shape=input_shape)
    # Feature normalization
    x = BatchNormalization()(inputs)

    # Temporal feature extraction
    x = Conv1D(64, kernel_size=3, activation='relu', padding='causal', kernel_regularizer = tf.keras.regularizers.l2(1e-4))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Hierarchical LSTM
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.4)(x)  # This was increased from 0.3 to 0.4
    x = LSTM(64)(x)

    # Simplified attention mechanism
    attention = Dense(64, activation='tanh')(x)
    attention = Dense(1, activation='sigmoid')(attention)
    x = Multiply()([x, attention])

    # Output - single scalar with sigmoid activation for binary classification
    output = Dense(1, activation='sigmoid', kernel_regularizer = tf.keras.regularizers.l2(1e-4)
)(x)

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