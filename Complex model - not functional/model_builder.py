"""multi-task model."""
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
    # Input layer with feature normalization
    inputs = Input(shape=input_shape)

    # Feature normalization layer (crucial for stability)
    normalized = BatchNormalization()(inputs)

    # First 1D Convolution for local pattern extraction
    conv1 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3,
        activation='relu', padding='same')(normalized)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)  # Added regularization

    #Second Conv Block, -> applied to the First Block
    conv1 = tf.keras.layers.Conv1D(
        filters=128, kernel_size=3,
        activation='relu', padding='same')(normalized)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)  # Added regularization


    # LSTM layers with attention
    lstm1 = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(conv1)  # Add recurrrent dropout
    attention = Attention(return_sequences=True)(lstm1)
    lstm2 = LSTM(64, return_sequences=False, recurrent_dropout=0.2)(attention)  # Add recurrrent dropout

    # Multi-task output heads with proper initialization
    volatility = Dense(1, activation='sigmoid', name='volatility',
                       kernel_initializer='glorot_uniform')(lstm2)
    trend = Dense(num_classes, activation='softmax', name='trend',
                  kernel_initializer='glorot_uniform')(lstm2)
    reversal = Dense(1, activation='sigmoid', name='reversal',
                     kernel_initializer='glorot_uniform')(lstm2)

    # Create model
    model = Model(inputs=inputs, outputs=[volatility, trend, reversal])

    # Custom loss weights
    loss_weights = {'volatility': 0.4, 'trend': 0.4, 'reversal': 0.2}

    # Custom metrics
    metrics = {
        'trend': ['accuracy', Precision(name='precision'), Recall(name='recall')],
        'reversal': ['accuracy', Precision(name='precision'), Recall(name='recall')]
    }

    # Compile model with gradient clipping
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipvalue=0.5),  # Gradient clipping
        loss={
            'volatility': 'mse',
            'trend': 'categorical_crossentropy',
            'reversal': 'binary_crossentropy'
        },
        loss_weights=loss_weights,
        metrics=metrics
    )

    print("✅ Model compiled with gradient clipping (clipvalue=0.5)")
    return model


def prepare_datasets(dataset, window_size=60, test_size=0.2):
    """Prepare datasets with strict data validation"""
    mandatory_features = [
        'open', 'high', 'low', 'close', 'returns', 'volatility',
        'rsi', 'macd', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
        'kst', 'squeeze', 'news_sentiment', 'volume'
    ]
    required_targets = ['vol_target', 'trend_target', 'reversal_target']

    all_X, all_y_vol, all_y_trend, all_y_reversal = [], [], [], []

    for symbol, data in dataset.items():
        print(f"\nProcessing {symbol}...")

        # Skip empty datasets
        if data.empty:
            print(f"⚠️ Empty data for {symbol} - skipping")
            continue

        # 1. Ensure all features exist
        processed_data = data.copy()
        for feature in mandatory_features + required_targets:
            if feature not in processed_data.columns:
                print(f"⚠️ Adding missing feature: {feature}")
                processed_data[feature] = 0

        # 2. Handle NaNs and Infs
        processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_data.dropna(subset=mandatory_features + required_targets, inplace=True)

        if processed_data.empty:
            print(f"⚠️ After NaN cleanup, {symbol} has no data - skipping")
            continue

        # 3. Validate target ranges
        # Volatility: Scale to [0,1] if needed
        vol_target = processed_data['vol_target'].values
        if vol_target.min() < 0 or vol_target.max() > 1:
            print(f"⚠️ Scaling volatility target for {symbol}")
            vmin, vmax = vol_target.min(), vol_target.max()
            processed_data['vol_target'] = (vol_target - vmin) / (vmax - vmin + 1e-8)

        # Trend: Only allow 0,1,2
        trend_target = processed_data['trend_target']
        valid_trend_mask = trend_target.isin([0, 1, 2])
        processed_data = processed_data[valid_trend_mask]

        # Reversal: Only allow 0 or 1
        reversal_target = processed_data['reversal_target']
        valid_reversal_mask = reversal_target.isin([0, 1])
        processed_data = processed_data[valid_reversal_mask]

        if processed_data.empty:
            print(f"⚠️ After target filtering, {symbol} has no data - skipping")
            continue

        # 4. Create sequences
        try:
            features = processed_data[mandatory_features]
            targets = processed_data[required_targets]

            # Scale features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)

            # Create sequences with safety check
            if len(scaled_features) < window_size:
                print(f"⚠️ Insufficient data for {symbol} (n={len(scaled_features)} < window_size) - skipping")
                continue

            X, yv = create_sequences(scaled_features, targets['vol_target'].values, window_size)
            _, yt = create_sequences(scaled_features, targets['trend_target'].values, window_size)
            _, yr = create_sequences(scaled_features, targets['reversal_target'].values, window_size)

            # One-hot encode trend
            encoder = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False)
            yt = encoder.fit_transform(yt.reshape(-1, 1))

            all_X.append(X)
            all_y_vol.append(yv)
            all_y_trend.append(yt)
            all_y_reversal.append(yr)

            print(f"✅ {symbol} processed | Samples: {X.shape[0]}")
            print(f"   Volatility range: [{yv.min():.4f}, {yv.max():.4f}]")
            print(f"   Trend distribution: {np.unique(yt.argmax(axis=1), return_counts=True)}")
            print(f"   Reversal distribution: {np.unique(yr, return_counts=True)}")

        except Exception as e:
            print(f"🔥 Processing failed for {symbol}: {str(e)}")
            continue

    # Validate combined data
    if not all_X:
        raise ValueError("❌ No valid datasets created")

    X_combined = np.concatenate(all_X)
    y_vol_combined = np.concatenate(all_y_vol)
    y_trend_combined = np.concatenate(all_y_trend)
    y_reversal_combined = np.concatenate(all_y_reversal)

    # Diagnostic logging
    print("\n🔍 Final dataset diagnostics:")
    print(f"Total samples: {X_combined.shape[0]}")
    print(f"Volatility range: [{y_vol_combined.min():.4f}, {y_vol_combined.max():.4f}]")
    print(f"Trend distribution: {np.unique(y_trend_combined.argmax(axis=1), return_counts=True)}")
    print(f"Reversal distribution: {np.unique(y_reversal_combined, return_counts=True)}")

    # Train-test split
    splits = train_test_split(
        X_combined,
        y_vol_combined,
        y_trend_combined,
        y_reversal_combined,
        test_size=test_size,
        shuffle=False  # Maintain temporal order
    )

    return (splits[0], [splits[2], splits[4], splits[6]]), \
        (splits[1], [splits[3], splits[5], splits[7]])


def train_model(model, train_data, val_data, epochs=100, batch_size=64):
    """Train model with enhanced callbacks"""
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs("models", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced from 10
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            filepath=os.path.join("models", f"best_model_{timestamp}.h5"),  # Use .h5 for compatibility
            save_best_only=True,
            monitor='val_loss'
        ),
        TensorBoard(
            log_dir=log_dir,
            profile_batch=0  # Disable profiling for stability
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Less aggressive reduction
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    print("\n🔥 Starting training with early stopping...")
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
        callbacks=callbacks,
        verbose=2
    )

    # Save final model
    model.save(os.path.join("models", f"final_model_{timestamp}.keras"))
    print(f"\n💾 Model saved to models/final_model_{timestamp}.keras")

    return model, history