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
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    Dense,
    Dropout,
    GlobalAveragePooling1D
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

# --- SHAP/plotting imports for feature explanation ---
import shap
import matplotlib.pyplot as plt
import numpy as np


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
        # # Feature engineering
        # data = generate_features(dataset, trading_type)
        # # Create target
        # # data['future_close'] = data.groupby('symbol')['close'].shift(-LOOKAHEAD_PERIOD) # for multiple symbols
        # data = create_direction_labels(data, lookahead=LOOKAHEAD_PERIOD, threshold=THRESHOLD)
        data = dataset.copy()
        # split data
        split_idx = int(len(data) * 0.70)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]

        # Define features - only use existing ones
        possible_features = []
        if trading_type == "forex":
            possible_features = [
                'open', 'high', 'low', 'close',
                # 'returns', 'momentum',
                # 'rsi', 'macd_line', 'macd_signal', 'macd_diff',
                # 'stoch',
                # 'atr',
                # 'atr_pct', 'bb_width',
                # 'hour', 'day_of_week', 'month'
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

        # Select feature columns and target separately
        X_train_raw = train_data[features].values
        y_train_raw = train_data['close'].values

        X_val_raw = val_data[features].values
        y_val_raw = val_data['close'].values

        # Scale features only (do NOT scale target)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        # Create sequences from scaled features and original target values
        X_train, y_train = create_sequences(X_train_scaled, y_train_raw, window_size)
        X_val, y_val = create_sequences(X_val_scaled, y_val_raw, window_size)

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
#     outputs = Dense(1, activation='linear')(x)
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005, clipnorm=0.1),
#         loss='mean_squared_error',
#         metrics=['mean_absolute_error', 'mean_squared_error']
#     )
#     return model


# def build_direction_model(input_shape):
#     inputs = Input(shape=input_shape)
#     x = BatchNormalization()(inputs)
#
#     # Add positional encoding (simple trainable)
#     positions = tf.range(start=0, limit=input_shape[0], delta=1)
#     pos_embedding = Embedding(input_shape[0], input_shape[1])(positions)
#     x = x + pos_embedding
#
#     # Transformer blocks
#     for _ in range(3):
#         x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3)
#
#     # Classification head
#     x = GlobalAveragePooling1D()(x)
#     x = Dropout(0.4)(x)
#     outputs = Dense(1, activation="linear")(x)
#
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005, clipnorm=0.1),
#         loss='mean_squared_error',
#         metrics=['mean_absolute_error', 'mean_squared_error']
#     )
#     return model


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


#
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
#     output = Dense(1, activation='linear')(x)
#     model = Model(inputs=inputs, outputs=output)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005, clipnorm=0.1),
#         loss='mean_squared_error',
#         metrics=['mean_absolute_error', 'mean_squared_error']
#     )
#     return model
#



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization + MHA
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs  # Residual connection

    # Feed Forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1], activation="linear")(x)
    return x + res  # Residual connection


import numpy as np
import tensorflow as tf

def get_positional_encoding(sequence_len, d_model):
    position = np.arange(sequence_len)[:, np.newaxis]  # (sequence_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model/2,)

    pe = np.zeros((sequence_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    pe = pe[np.newaxis, ...]  # shape (1, sequence_len, d_model)
    return tf.cast(pe, dtype=tf.float32)

def build_direction_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)

    # Generate positional encoding tensor once
    pos_encoding = get_positional_encoding(input_shape[0], input_shape[1])
    # Add positional encoding to inputs
    x = x + pos_encoding

    for _ in range(3):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)  # Regression output

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=0.1),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, trading_type, asset_name, epochs,
                batch_size):
    """Train directional model with callbacks"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("models", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6)
    ]

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


feature_names = [
    'open', 'high', 'low', 'close', 'returns',
    'momentum',
    'rsi', 'macd_line', 'macd_signal', 'macd_diff',
    # 'stoch',
    'atr',
    'atr_pct', 'bb_width',
    'hour', 'day_of_week', 'month']


# --- SHAP feature importance explanation function ---
def explain_features_with_shap(model, X_sample):
    """
    Explain model predictions with SHAP values.
    :param model: Trained Keras model
    :param X_sample: A numpy array (samples, seq_len, features) to explain
    :param feature_names: List of feature names in order
    """

    # Use a small background dataset for speed (e.g. 100 samples)
    background = X_sample[np.random.choice(X_sample.shape[0], min(100, X_sample.shape[0]), replace=False)]

    # Create DeepExplainer
    explainer = shap.DeepExplainer(model, background)

    # Compute SHAP values for a subset (e.g. first 50 samples)
    samples_to_explain = X_sample[:50]
    shap_values = explainer.shap_values(samples_to_explain)

    # Plot SHAP summary for class 0 (adjust if needed for multi-class)
    shap.summary_plot(shap_values[0], samples_to_explain, feature_names=feature_names)
