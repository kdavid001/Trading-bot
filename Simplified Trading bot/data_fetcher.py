import pandas as pd
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from twelvedata import TDClient
from config import CONFIG
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        self.api_status = {
            'alpha_vantage': False,
            'twelvedata': False,
            'newsapi': False,
            'yfinance': True  # Always available
        }
        try:
            self.av_fx = ForeignExchange(CONFIG.get('alpha_vantage', '')) if CONFIG.get('alpha_vantage') else None
            self.av_crypto = CryptoCurrencies(CONFIG.get('alpha_vantage', '')) if CONFIG.get('alpha_vantage') else None
            if self.av_fx:
                self.api_status['alpha_vantage'] = True
        except Exception as e:
            logger.warning(f"Alpha Vantage initialization failed: {e}")

        try:
            self.td = TDClient(apikey=CONFIG.get('twelvedata', '')) if CONFIG.get('twelvedata') else None
            if self.td:
                self.api_status['twelvedata'] = True
        except Exception as e:
            logger.warning(f"TwelveData initialization failed: {e}")

        self.last_api_call = time.time()
        self.min_call_interval = 5  # Reduced from 15 to 5 seconds
        self.max_historical_points = 5000
        self.yfinance_fallback = True

    def _rate_limit(self):
        """Enforce API rate limits with better feedback"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_interval:
            wait_time = self.min_call_interval - elapsed
            logger.info(f"Rate limiting: Waiting {wait_time:.1f} seconds before next API call")
            time.sleep(wait_time)
        self.last_api_call = time.time()

    def _clean_data(self, data, trading_type):
        """Unified data cleaning and standardization for different trading types"""
        df = data.copy()

        # Convert index to datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df = df.set_index('date')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    df = df.reset_index(drop=True)
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    df = df.set_index('datetime')

        # Standardize column names
        col_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'close',
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure numeric conversion
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Determine required columns based on trading type
        if trading_type == 'forex':
            required_cols = ['open', 'high', 'low', 'close']
        elif trading_type == 'crypto':
            required_cols = ['open', 'high', 'low', 'close', 'volume']
        else:
            print(f"Trading type {trading_type} not supported\n")
            print("Defaulting to original OHLCV")
            required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Keep only available required columns
        available_cols = [col for col in required_cols if col in df.columns]

        # Forward fill and drop NA
        df = df[available_cols].sort_index().ffill().dropna()

        return df

    def get_market_data(self, symbol, lookback_years, trading_type, interval='daily'):
        """
        Fetch OHLCV data with historical depth
        Args:
            symbol: Asset symbol (e.g., 'BTC/USD', 'EUR/USD', 'AAPL')
            interval: Time interval ('daily', 'weekly', 'monthly', '15min', etc.)
            lookback_years: Years of historical data to retrieve
        """
        try:
            # Use yfinance as the primary source for historical data
            if interval in ['daily', 'weekly', 'monthly']:
                return self._get_yfinance_data(symbol, interval, lookback_years, trading_type)
            else:
                return self._get_intraday_data(symbol, interval, trading_type)
        except Exception as e:
            print(f"❌ yfinance failed: {e}. Falling back to other APIs...")
            return self._get_data_with_fallback(symbol, interval, lookback_years, trading_type)

    def _get_data_with_fallback(self, symbol, interval, lookback_years, trading_type):
        """Fallback method when yfinance fails"""
        try:
            # For daily/weekly/monthly data
            if interval in ['daily', 'weekly', 'monthly']:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_years * 365)

                # Try TwelveData if available
                if self.td:
                    try:
                        # Convert interval to TwelveData format
                        td_interval = {
                            'daily': '1day',
                            'weekly': '1week',
                            'monthly': '1month'
                        }[interval]

                        # Format symbol for TwelveData
                        if '/' in symbol:
                            formatted_symbol = symbol.replace('/', '')
                        else:
                            formatted_symbol = symbol

                        data = self.td.time_series(
                            symbol=formatted_symbol,
                            interval=td_interval,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            outputsize=min(self.max_historical_points, lookback_years * 365)
                        ).as_pandas()

                        if not data.empty:
                            return self._clean_data(data, trading_type)
                    except Exception as e:
                        print(f"⚠️ TwelveData failed: {e}")

                # Try Alpha Vantage for crypto/USD pairs
                if '/' in symbol and self.av_crypto:
                    base, quote = symbol.split('/')
                    if quote == 'USD' and interval == 'daily':
                        try:
                            data, _ = self.av_crypto.get_digital_currency_daily(
                                symbol=base, market='USD'
                            )
                            return self._clean_data(pd.DataFrame(data).transpose(), trading_type)
                        except Exception as e:
                            print(f"⚠️ Alpha Vantage failed: {e}")

            # For intraday data
            return self._get_intraday_data(symbol, interval, trading_type)

        except Exception as e:
            print(f"❌ All data sources failed for {symbol}: {e}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def _get_yfinance_data(self, symbol, interval, years, trading_type):
        """Fetch data using yfinance (most reliable for historical data)"""
        # Convert symbol to yfinance format
        if '/' in symbol:
            base, quote = symbol.split('/')
            # Crypto pairs
            if base.upper() in ['BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'ADA', 'DOT', 'LINK', 'BNB', 'XLM']:
                yf_symbol = f"{base}-{quote}"
            # Forex pairs
            else:
                yf_symbol = f"{base}{quote}=X"
        else:
            yf_symbol = symbol

        # Map intervals to yfinance format
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo',
            '15min': '15m',
            '1h': '60m',
            '4h': '4h'
        }
        yf_interval = interval_map.get(interval.lower(), interval.lower())

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        # FIX: Use the Ticker API instead of download for better compatibility
        try:
            ticker = yf.Ticker(yf_symbol)

            # Use history method instead of download
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True
            )
        except Exception as e:
            # Fallback to download method if history fails
            try:
                data = yf.download(
                    yf_symbol,
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True
                )
            except:
                # Final fallback without auto_adjust
                data = yf.download(
                    yf_symbol,
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    progress=False
                )

        if data.empty:
            raise ValueError(f"yfinance returned empty dataset for {yf_symbol}")

        return self._clean_data(data, trading_type)

    def _get_intraday_data(self, symbol, interval, trading_type):
        """Fetch intraday data with point-based lookback"""
        try:
            # First try TwelveData
            if self.td:
                # Convert interval to TwelveData format
                td_interval_map = {
                    '15min': '15min',
                    '1h': '1h',
                    '4h': '4h',
                    'daily': '1day'
                }
                td_interval = td_interval_map.get(interval.lower(), interval.lower())

                # Format symbol for TwelveData
                if '/' in symbol:
                    formatted_symbol = symbol.replace('/', '')
                else:
                    formatted_symbol = symbol

                return self.td.time_series(
                    symbol=formatted_symbol,
                    interval=td_interval,
                    outputsize=self.max_historical_points
                ).as_pandas()

            # Fallback to Alpha Vantage for forex
            elif '/' in symbol and self.av_fx:
                base, quote = symbol.split('/')
                # Convert interval to AlphaVantage format
                av_interval_map = {
                    '15min': '15min',
                    '1h': '60min',
                    '4h': '240min'
                }
                av_interval = av_interval_map.get(interval.lower(), '15min')

                data, _ = self.av_fx.get_currency_exchange_intraday(
                    from_symbol=base,
                    to_symbol=quote,
                    interval=av_interval,
                    outputsize='full'
                )
                return self._clean_data(pd.DataFrame(data).transpose(), trading_type)

            # Fallback to yfinance
            else:
                return self._get_yfinance_data(symbol, interval, 1, trading_type)  # 1 year max for intraday

        except Exception as e:
            print(f"⚠️ Intraday fetch failed: {e}")
            return self._get_yfinance_data(symbol, interval, 1, trading_type)



# example usage
# if __name__ == "__main__":
#     fetcher = DataFetcher()
#
#     print("\n=== Historical Data Test ===")
#     print("BTC/USD (5 years daily):")
#     btc_data = fetcher.get_market_data('BTC/USD', interval='daily', lookback_years=5)
#     print(f"Retrieved {len(btc_data)} daily bars")
#     if not btc_data.empty:
#         print(f"From {btc_data.index[0].date()} to {btc_data.index[-1].date()}")
#
#     print("\nEUR/USD (3 years weekly):")
#     eur_data = fetcher.get_market_data('EUR/USD', interval='weekly', lookback_years=3)
#     print(f"Retrieved {len(eur_data)} weekly bars")
#     if not eur_data.empty:
#         print(f"From {eur_data.index[0].date()} to {eur_data.index[-1].date()}")
#
#     print("\n=== Intraday Data Test ===")
#     print("SPY (15min intervals):")
#     spy_data = fetcher.get_market_data('SPY', interval='15min')
#     print(f"Retrieved {len(spy_data)} intraday bars")
#
#     print("\n=== News Test ===")
#     news = fetcher.get_news('stock market', lookback_days=3)
#     print(f"Retrieved {len(news)} news articles")
#     if not news.empty:
#         print(f"First article: {news.iloc[3]['title']}")
#
