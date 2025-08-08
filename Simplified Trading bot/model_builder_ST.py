import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, AUC


# # Configuration
# Shorter window(30–60) → Captures short - term momentum,
# reacts fast to news, but may miss longer patterns.

# Longer window(120–240) → Captures trend context, but may
# introduce noise for short - term predictions and require more data per sample.


def create_sequences(data, targets, window_size):
    """Create time-series sequences for LSTM forecasting"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])  # Input window: [t-w, t-1]
        y.append(targets[i])  # Target: value at t (next step)
    return np.array(X), np.array(y)


# def create_sequences(data, feature_cols, targets, seq_len=60, date_col='Date', dropna=True):
#     """
#     Create sequences for time-series modeling with explicit date handling.
#
#     Args:
#         df (pd.DataFrame): DataFrame with time-series data.
#         feature_cols (list): Columns to use as features.
#         target_col (str): Column to use as target.
#         seq_len (int): Length of each input sequence.
#         date_col (str): Name of the datetime column.
#         dropna (bool): If True, drop sequences with missing values.
#
#     Returns:
#         np.ndarray: Feature sequences (samples, seq_len, features)
#         np.ndarray: Target sequences (samples),
#         np.ndarray: Corresponding last date of each sequence (samples,)
#     """
#     # Ensure sorted by date
#     df = data.sort_values(date_col)
#     X, y, dates = [], [], []
#
#     for i in range(len(df) - seq_len):
#         seq = df.iloc[i:i+seq_len]
#         target = df.iloc[i+seq_len][targets]
#         last_date = seq.iloc[-1][date_col]
#
#         if dropna and (seq[feature_cols].isnull().any().any() or pd.isnull(target)):
#             continue
#
#         X.append(seq[feature_cols].values)
#         y.append(target)
#         dates.append(last_date)
#
#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(dates)

def generate_features(data, trading_type):
    """Safe feature engineering with fallbacks for all features"""
    data = data.copy()
    # 1. Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = set(required_cols) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Calculate basic features with shift(1) to prevent lookahead
    # if you have other trading techniques you'd like to add do it here
    data['returns'] = data['close'].pct_change().shift(1)  # calculates the returns of 1 row compared to the other (
    # Close)
    data['volatility'] = (data['high'] - data['low']).shift(1)
    data['momentum'] = data['close'].pct_change(5).shift(1)

    # 3. Volume features (with fallback)
    if trading_type == 'crypto':
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
        # data['vol_regime'] = 0
        pass

    # 5. Only drop rows where essential features are missing
    # essential = ['returns', 'volatility', 'momentum']
    essential = ['returns']
    return data.dropna(subset=essential)


def create_direction_labels(data, lookahead, threshold):
    data['future_close'] = data['close'].shift(-lookahead)
    pct_change = (data['future_close'] - data['close']) / data['close']

    data['direction'] = np.where(
        pct_change > threshold, 1,
        np.where(pct_change < -threshold, -1, 0)
    )
    return data


def prepare_dataset(dataset, trading_type, LOOKAHEAD_PERIOD, THRESHOLD, window_size):
    """More robust dataset preparation"""
    try:
        # Feature engineering
        data = generate_features(dataset, trading_type)
        # Create target
        # data['future_close'] = data.groupby('symbol')['close'].shift(-LOOKAHEAD_PERIOD) # for multiple symbols
        data = create_direction_labels(data, lookahead=LOOKAHEAD_PERIOD, threshold=THRESHOLD)

        # split data
        split_idx = int(len(data) * 0.70)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]

        # Define features - only use existing ones
        possible_features = []
        if trading_type == "forex":
            possible_features = [
                'open', 'high', 'low', 'close', 'returns',
                'momentum',
                'rsi', 'macd_line', 'macd_signal', 'macd_diff',
                # 'stoch',
                'atr',
                'atr_pct', 'bb_width',
                'hour', 'day_of_week', 'month'
                # 'kst', 'squeeze'
            ]
        elif trading_type == "crypto":
            possible_features = [
                'open', 'high', 'low', 'close', 'returns', 'volatility', 'momentum',
                'volume_z', 'rsi', 'macd_line', 'macd_signal', 'macd_diff', 'stoch',
                'atr', 'atr_pct', 'bb_width', 'hour', 'day_of_week', 'month',
                'kst', 'squeeze', 'vol_regime'
            ]
        features = [f for f in possible_features if f in data.columns]

        # scaling for forex done to focus on multiple symbols just incase.
        scalers = {}
        train_scaled = []
        if trading_type == 'crypto':
            for symbol, group in train_data.groupby('symbol'):
                scaler = MinMaxScaler()
                scaled = group.copy()
                scaled[features] = scaler.fit_transform(group[features])
                scalers[symbol] = scaler
                train_scaled.append(scaled)


        elif trading_type == 'forex':
            for symbol, group in train_data.groupby('symbol'):
                scaler = MinMaxScaler()
                scaled = group.copy()
                scalers[symbol] = scaler
                # scaled[features] = scaler.fit_transform(group[features])
                scaled[features] = scaler.fit_transform(group[features].fillna(0))
                train_scaled.append(scaled)

        train_scaled = pd.concat(train_scaled)
        train_scaled["Date"] = train_scaled.index

        # print("Train class distribution:", train_data[features].value_counts())
        print("Validation class distribution:", val_data['direction'].value_counts())

        # train_scaled.to_csv("data/scaled_data.csv", index=True)
        # print("✅ Saved  scaled date to combined_data_pipeline.csv")

        # Scale validation data
        val_scaled = []
        for symbol, group in val_data.groupby('symbol'):
            if symbol in scalers:
                scaled = group.copy()
                scaled[features] = scalers[symbol].transform(group[features])
                val_scaled.append(scaled)
        val_scaled = pd.concat(val_scaled)

        # print(f"These are the Column for the trained data: {train_scaled[features].columns}")
        # print(f"These are the Column for the Val_data: {val_scaled[features].columns}")

        # train_scaled[features].to_csv("data/train_scaled.csv", index=True)
        # print("saved x_val")
        # train_scaled["direction"].to_csv("data/train_scaled_direction.csv", index=True)
        # print("saved directions")


        # Create sequences
        X_train, y_train = create_sequences(train_scaled[features].values, train_scaled['direction'].values, window_size)
        X_val, y_val = create_sequences(val_scaled[features].values, val_scaled['direction'].values, window_size)

        return (X_train, y_train), (X_val, y_val)

    except Exception as e:
        print(f"Error in prepare_dataset: {str(e)}")
        # Return empty arrays if something fails
        empty = np.array([])
        return (empty, empty), (empty, empty)

# working model
# def build_direction_model(input_shape):
#     inputs = Input(shape=input_shape)
#     x = BatchNormalization()(inputs)
#     x = Conv1D(64, 3, activation='relu', padding='causal')(x)
#     x = LSTM(64, return_sequences=True)(x)
#     x = LSTM(32)(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     outputs = Dense(3, activation='softmax')(x)
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005, clipnorm=0.1),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


def build_direction_model(input_shape):
    """Build enhanced model for directional prediction"""
    inputs = Input(shape=input_shape)
    # Feature normalization
    x = BatchNormalization()(inputs)

    # Temporal feature extraction
    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # Hierarchical LSTM
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.4)(x)
    x = LSTM(32)(x)

    # Simplified attention mechanism
    attention = Dense(32, activation='tanh')(x)
    attention = Dense(1, activation='sigmoid')(attention)
    x = Multiply()([x, attention])

    output = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-3)
                   )(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
            optimizer=Adam(learning_rate=0.0005, clipnorm=0.1),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    return model


# def build_direction_model(input_shape):
#     inputs = Input(shape=input_shape)
#
#     # Feature-wise normalization
#     x = BatchNormalization(axis=-1)(inputs)
#
#     # Temporal convolution blocks with feature-aware kernels
#     x = Conv1D(64, kernel_size=5, activation='relu', padding='causal',
#                kernel_regularizer=l2(1e-4))(x)
#     x = Conv1D(128, kernel_size=3, activation='relu', padding='causal',
#                kernel_regularizer=l2(1e-4))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)
#
#     # Feature-reduction convolution
#     x = Conv1D(64, kernel_size=1, activation='relu')(x)  # Reduce feature dimensions
#
#     # Depthwise separable convolution for efficiency
#     x = Conv1D(128, kernel_size=3, activation='relu', padding='same',
#                groups=8, kernel_regularizer=l2(1e-4))(x)
#
#     # Hierarchical LSTM processing
#     x = LSTM(128, return_sequences=True)(x)
#     x = LSTM(64, return_sequences=False)(x)
#
#     # Feature refinement
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.5)(x)
#
#     # Output layer
#     output = Dense(1, activation='sigmoid')(x)
#
#     model = Model(inputs=inputs, outputs=output)
#
#     # Optimizer with warmup
#     opt = Adam(learning_rate=0.00015)
#
#     model.compile(
#         optimizer=opt,
#         loss='binary_crossentropy',
#         metrics=['accuracy', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')]
#     )
#
#     return model


#working model
# def build_direction_model(input_shape):
#     inputs = Input(shape=input_shape)
#     x = BatchNormalization()(inputs)
#     x = Conv1D(64, 3, activation='relu', padding='causal')(x)
#     x = LSTM(64, return_sequences=True)(x)
#     x = LSTM(32)(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     outputs = Dense(3, activation='softmax')(x)
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005, clipnorm=0.1),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# def build_direction_model(input_shape):
#     inputs = Input(shape=input_shape)
#     # Feature-wise normalization
#     x = BatchNormalization(axis=-1)(inputs)
#     # Temporal convolution blocks with feature-aware kernels
#     x = Conv1D(64, kernel_size=5, activation='relu', padding='causal',
#                kernel_regularizer=l2(1e-4))(x)
#     x = Conv1D(128, kernel_size=3, activation='relu', padding='causal',
#                kernel_regularizer=l2(1e-4))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)
#
#     # Feature-reduction convolution
#     x = Conv1D(64, kernel_size=1, activation='relu')(x)  # Reduce feature dimensions
#
#     # Depthwise separable convolution for efficiency
#     x = Conv1D(128, kernel_size=3, activation='relu', padding='same',
#                groups=8, kernel_regularizer=l2(1e-4))(x)
#     # Hierarchical LSTM processing
#     # x = LSTM(128, return_sequences=True)(x)
#     x = LSTM(64, return_sequences=True)(x)
#     x = LSTM(32, return_sequences=False)(x)
#     # Feature refinement
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     # Output layer
#     output = Dense(3, activation='softmax')(x)
#     model = Model(inputs=inputs, outputs=output)
#     # Optimizer with warmup
#     opt = Adam(learning_rate=0.0005)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005, clipnorm=0.1),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model





def train_model(model, X_train, y_train, X_val, y_val, trading_type, class_weight, asset_name, epochs,
                batch_size):
    """Train directional model with callbacks"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("models", exist_ok=True)

    callbacks = [
        EarlyStopping(patience=8, mode="max", min_delta=0.01, restore_best_weights=True),
        ModelCheckpoint(
            f"models/best_model_checkpoint:{timestamp}.h5",
            save_best_only=True,
            monitor='val_loss'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,
            min_lr=1e-6,
            cooldown=2,
        )
    ]

    print("\n🔥 Training directional model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Save final model
    model.save(f"models/direction_model_{trading_type}_{timestamp}_{asset_name}.keras")
    print(f"\n💾 Model saved to: models/direction_model_{trading_type}_{timestamp}_{asset_name}.keras")
    return model, history
