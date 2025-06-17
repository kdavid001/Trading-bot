import pandas as pd
import ta
import pandas_ta as pta
import numpy as np
from typing import Optional


class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators with robust error handling
        Returns: DataFrame with technical indicators
        """
        # Validate input
        if df.empty:
            return pd.DataFrame()

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols} - skipping feature engineering")
            return pd.DataFrame()

        df = df.copy()

        try:
            # 1. Basic price features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20, min_periods=1).std()

            # 2. Volume indicators (skip if volume missing)
            if 'volume' in df.columns and df['volume'].notna().any():
                try:
                    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                    df['cmf'] = ta.volume.chaikin_money_flow(
                        df['high'], df['low'], df['close'], df['volume'], 20)
                except Exception as e:
                    print(f"⚠️ Volume indicator error: {str(e)}")
                    df['obv'] = np.nan
                    df['cmf'] = np.nan
            else:
                df['obv'] = np.nan
                df['cmf'] = np.nan

            # 3. Core indicators (must work)
            df['rsi'] = ta.momentum.rsi(df['close'], 14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], 14)

            # 4. Bollinger Bands
            try:
                bb = ta.volatility.BollingerBands(df['close'])
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_middle'] = bb.bollinger_mavg()
                df['bb_lower'] = bb.bollinger_lband()
            except Exception as e:
                print(f"⚠️ Bollinger Bands error: {str(e)}")
                df['bb_upper'] = np.nan
                df['bb_middle'] = np.nan
                df['bb_lower'] = np.nan

            # 5. Advanced indicators
            try:
                df['kst'] = pta.kst(df['close'])['KST_10_15_20_30_10_10_10_15']
                df['squeeze'] = pta.squeeze(df['high'], df['low'], df['close'])['SQZ_20_2.0_20_1.5']
            except Exception as e:
                print(f"⚠️ Advanced indicator error: {str(e)}")
                df['kst'] = np.nan
                df['squeeze'] = np.nan

            # Only drop rows where essential indicators are missing
            essential_cols = ['returns', 'volatility', 'rsi', 'macd', 'atr']
            return df.dropna(subset=essential_cols, how='all')

        except Exception as e:
            print(f"🔥 Feature engineering failed: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def create_targets(df: pd.DataFrame, future_bars: int = 3) -> pd.DataFrame:
        """
        Creates ML targets with validation
        Args:
            future_bars: Number of bars ahead to predict
        Returns: DataFrame with targets or empty DataFrame if invalid
        """
        if df.empty or len(df) < future_bars + 5:
            print("⚠️ Not enough data for target creation")
            return pd.DataFrame()

        df = df.copy()

        try:
            # Volatility target
            if 'atr' in df.columns:
                df['vol_target'] = df['atr'].shift(-future_bars).rolling(5, min_periods=1).mean()
            else:
                df['vol_target'] = np.nan

            # Trend target
            df['trend_target'] = (df['close'].shift(-future_bars) > df['close']).astype(int)

            # Reversal target
            if 'rsi' in df.columns:
                df['reversal_target'] = ((df['rsi'] > 70) | (df['rsi'] < 30)).astype(int)
            else:
                df['reversal_target'] = 0

            return df.dropna(subset=['trend_target', 'reversal_target'])

        except Exception as e:
            print(f"🔥 Target creation failed: {str(e)}")
            return pd.DataFrame()