import pandas as pd
import ta
import pandas_ta as pta
import numpy as np
from typing import Optional, Dict, List


class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced technical feature engineering with:
        - More robust calculations
        - Additional feature types
        - Better normalization
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
            # 1. Price Transformations
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['close'].rolling(20).std()

            # 2. Volume Features
            if 'volume' in df.columns:
                df['volume_pct'] = df['volume'].pct_change()
                df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                df['volume_z'] = FeatureEngineer._zscore(df['volume'], window=20)

            # 3. Momentum Indicators
            df['momentum'] = df['close'].pct_change(5)
            df['rsi'] = ta.momentum.rsi(df['close'], 14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])

            # 4. Volatility Features
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
            df['atr_pct'] = df['atr'] / df['close']
            df['bb_width'] = (ta.volatility.bollinger_hband(df['close']) -
                              ta.volatility.bollinger_lband(df['close'])) / df['close']

            # 5. Time Features
            if isinstance(df.index, pd.DatetimeIndex):
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month

            # 6. Advanced Features
            try:
                df['kst'] = pta.kst(df['close'])['KST_10_15_20_30_10_10_10_15']
                df['squeeze'] = pta.squeeze(df['high'], df['low'], df['close'])['SQZ_ON']
            except Exception as e:
                print(f"Advanced indicators failed: {e}")

            # Cleanup
            essential = ['returns', 'volatility', 'rsi', 'macd', 'atr']
            return df.dropna(subset=essential, how='all')

        except Exception as e:
            print(f"Feature engineering failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Robust z-score calculation"""
        mean = series.rolling(window, min_periods=5).mean()
        std = series.rolling(window, min_periods=5).std()
        return (series - mean) / (std.replace(0, 1e-8))

    @staticmethod
    def create_targets(df: pd.DataFrame,
                       future_bars: int = 3,
                       threshold: float = 0.0015) -> pd.DataFrame:
        """Enhanced target creation with triple barrier method"""
        if df.empty or len(df) < future_bars + 5:
            return pd.DataFrame()

        df = df.copy()

        try:
            future_close = df['close'].shift(-future_bars)

            # Trend target (-1, 0, 1)
            df['trend_target'] = np.select(
                [
                    future_close > df['close'] * (1 + threshold),
                    future_close < df['close'] * (1 - threshold)
                ],
                [1, -1],
                default=0
            )

            # Volatility target
            if 'atr' in df.columns:
                df['vol_target'] = df['atr'].shift(-future_bars).rolling(5).mean()

            # Reversal probability
            if 'rsi' in df.columns:
                df['reversal_prob'] = (
                        (df['rsi'].rolling(3).max() > 70) |
                        (df['rsi'].rolling(3).min() < 30)
                ).astype(int)

            return df.dropna(subset=['trend_target'])

        except Exception as e:
            print(f"Target creation error: {e}")
            return pd.DataFrame()

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