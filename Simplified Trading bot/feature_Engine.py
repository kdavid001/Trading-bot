import pandas as pd
import ta
import pandas_ta as pta
import numpy as np
from typing import Optional, Dict, List

import pandas as pd
import ta
import pandas_ta as pta
import numpy as np
from typing import Dict, List


class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Robust technical feature engineering with error handling
        """
        # Validation
        if df.empty:
            return pd.DataFrame()

        required_cols = ['open', 'high', 'low', 'close']
        if not set(required_cols).issubset(df.columns):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns: {missing}")
            return pd.DataFrame()

        df = df.copy()

        try:
            # 1. Price Transformations (safe calculations)
            df['returns'] = df['close'].pct_change().shift(1)
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).shift(1)
            df['volatility'] = df['close'].rolling(20).std().shift(1)

            # 2. Volume Features (with validation)
            if 'volume' in df.columns and not df['volume'].isnull().all():
                df['volume_pct'] = df['volume'].pct_change().shift(1)
                vol_ma = df['volume'].rolling(20).mean().shift(1)
                df['volume_ma_ratio'] = (df['volume'] / vol_ma).replace([np.inf, -np.inf], 1)
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume']).shift(1)
                df['volume_z'] = FeatureEngineer._zscore(df['volume'], window=20).shift(1)
            else:
                df['volume'] = 0
                df['volume_pct'] = 0
                df['volume_ma_ratio'] = 1
                df['obv'] = 0
                df['volume_z'] = 0

            # 3. Momentum Indicators (with MACD fix)
            df['momentum'] = df['close'].pct_change(5).shift(1)
            df['rsi'] = ta.momentum.rsi(df['close'], 14).shift(1)

            # Fixed MACD calculation
            try:
                macd = ta.trend.MACD(df['close'])
                df['macd_line'] = macd.macd().shift(1)
                df['macd_signal'] = macd.macd_signal().shift(1)
                df['macd_diff'] = macd.macd_diff().shift(1)
            except Exception as e:
                print(f"MACD calculation failed: {e}")
                df['macd_line'] = 0
                df['macd_signal'] = 0
                df['macd_diff'] = 0

            df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close']).shift(1)

            # 4. Volatility Features
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], 14
            ).shift(1)
            df['atr_pct'] = (df['atr'] / df['close']).shift(1)

            # Bollinger Bands with error handling
            try:
                hband = ta.volatility.bollinger_hband(df['close'])
                lband = ta.volatility.bollinger_lband(df['close'])
                df['bb_width'] = ((hband - lband) / df['close']).shift(1)
            except:
                df['bb_width'] = 0

            # 5. Time Features (safe)
            if isinstance(df.index, pd.DatetimeIndex):
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month

            # 6. Advanced Features (with fallbacks)
            try:
                kst = pta.kst(df['close'])
                df['kst'] = kst['KST_10_15_20_30_10_10_10_15'].shift(1)
            except:
                df['kst'] = df['close'].pct_change(10).shift(1)  # Simple momentum fallback

            try:
                squeeze = pta.squeeze(df['high'], df['low'], df['close'])
                df['squeeze'] = squeeze['SQZ_ON'].shift(1)
            except:
                df['squeeze'] = 0

            # Cleanup - require fewer essential features
            essential = ['returns', 'rsi', 'atr']
            return df.dropna(subset=essential)

        except Exception as e:
            print(f"Feature engineering failed: {str(e)}")
            return pd.DataFrame()



    @staticmethod
    def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Robust z-score calculation"""
        mean = series.rolling(window, min_periods=5).mean()
        std = series.rolling(window, min_periods=5).std()
        return (series - mean) / (std.replace(0, 1e-8))

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> bool:
        """Validate input DataFrame structure"""
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Error: DataFrame must have DatetimeIndex")
            return False
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            print(f"Missing required columns: {set(required) - set(df.columns)}")
            return False
        return True

    @staticmethod
    def create_targets(df: pd.DataFrame, future_bars: int = 3, threshold: float = 0.0015):
        """More robust target creation with dynamic thresholds"""
        if not FeatureEngineer._validate_input(df):
            return pd.DataFrame()

        df = df.copy()

        # Dynamic threshold based on recent volatility
        if 'atr' in df.columns:
            threshold = df['atr'].rolling(14).mean().shift(1) / df['close'] * 2

        # Triple barrier method
        future_high = df['high'].rolling(future_bars).max().shift(-future_bars)
        future_low = df['low'].rolling(future_bars).min().shift(-future_bars)

        df['target'] = np.select(
            [
                future_high > df['close'] * (1 + threshold),
                future_low < df['close'] * (1 - threshold)
            ],
            [1, -1],
            default=0
        )

        # Add meta-features for model interpretability
        df['target_strength'] = np.where(
            df['target'] != 0,
            np.abs(future_high / df['close'] - 1) if df['target'] > 0
            else np.abs(1 - future_low / df['close']),
            0
        )

        return df.dropna(subset=['target'])

    @staticmethod
    def get_feature_categories() -> Dict[str, List[str]]:
        """Returns organized feature categories"""
        return {
            'price': ['open', 'high', 'low', 'close', 'returns', 'log_returns'],
            'volume': ['volume', 'volume_pct', 'volume_ma_ratio', 'obv', 'volume_z'],
            'momentum': ['momentum', 'rsi', 'macd', 'stoch', 'kst'],
            'volatility': ['volatility', 'atr', 'atr_pct', 'bb_width', 'squeeze'],
            'time': ['hour', 'day_of_week', 'month']
        }

    @staticmethod
    def get_feature_importance() -> Dict[str, float]:
        """Estimated predictive value of features"""
        return {
            'rsi': 0.85,
            'macd_diff': 0.78,
            'atr_pct': 0.72,
            'volume_z': 0.65,
            'bb_width': 0.63,
            'kst': 0.60,
            'squeeze': 0.58
        }