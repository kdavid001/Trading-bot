# data_fetcher.py
import pandas as pd
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from twelvedata import TDClient
from newsapi import NewsApiClient
from config import CONFIG
import time
from datetime import datetime, timedelta


class DataFetcher:
    def __init__(self):
        # Initialize APIs with rate limiting protection
        self.av_fx = ForeignExchange(CONFIG.get('alpha_vantage', '')) if CONFIG.get('alpha_vantage') else None
        self.av_crypto = CryptoCurrencies(CONFIG.get('alpha_vantage', '')) if CONFIG.get('alpha_vantage') else None
        self.td = TDClient(apikey=CONFIG.get('twelvedata', '')) if CONFIG.get('twelvedata') else None
        self.newsapi = NewsApiClient(api_key=CONFIG.get('newsapi', '')) if CONFIG.get('newsapi') else None

        self.last_api_call = time.time()
        self.min_call_interval = 15  # Seconds between API calls
        self.max_historical_points = 5000  # Max data points per API request
        self.yfinance_fallback = True  # Use yfinance when other APIs fail

    def _rate_limit(self):
        """Enforce API rate limits"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_api_call = time.time()

    def get_market_data(self, symbol, interval='daily', lookback_years=5):
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
                return self._get_yfinance_data(symbol, interval, lookback_years)
            else:
                return self._get_intraday_data(symbol, interval)
        except Exception as e:
            print(f"❌ yfinance failed: {e}. Falling back to other APIs...")
            return self._get_data_with_fallback(symbol, interval, lookback_years)

    def _get_data_with_fallback(self, symbol, interval, lookback_years):
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
                            return self._clean_data(data)
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
                            return self._clean_data(pd.DataFrame(data).transpose())
                        except Exception as e:
                            print(f"⚠️ Alpha Vantage failed: {e}")

            # For intraday data
            return self._get_intraday_data(symbol, interval)

        except Exception as e:
            print(f"❌ All data sources failed for {symbol}: {e}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def _get_yfinance_data(self, symbol, interval, years):
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
            '4h': '4h',
            'daily': '1d'
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

        return self._clean_data(data)

    def _get_intraday_data(self, symbol, interval):
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
                return self._clean_data(pd.DataFrame(data).transpose())

            # Fallback to yfinance
            else:
                return self._get_yfinance_data(symbol, interval, 1)  # 1 year max for intraday

        except Exception as e:
            print(f"⚠️ Intraday fetch failed: {e}")
            return self._get_yfinance_data(symbol, interval, 1)

    def _clean_data(self, data):
        """Standardize and clean financial data"""
        if data.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        # Create copy to avoid SettingWithCopyWarning
        df = data.copy()

        # Convert index to datetime
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except:
            try:
                df.reset_index(inplace=True)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif 'index' in df.columns:
                    df['index'] = pd.to_datetime(df['index'])
                    df.set_index('index', inplace=True)
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                df = df.sort_index()
            except:
                df = df.reset_index(drop=True)

        # Handle different column naming conventions
        column_map = {
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
            '5. volume': 'volume',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Forward-fill missing values and drop any remaining NaNs
        df = df.ffill().dropna()

        # Ensure we have the required columns
        return df[['open', 'high', 'low', 'close', 'volume']] if all(col in df.columns for col in numeric_cols) else df

    def get_news(self, query, lookback_days=7):
        """Fetch financial news articles with improved error handling"""
        if not self.newsapi:
            print("⚠️ NewsAPI not configured")
            return []

        try:
            self._rate_limit()
            from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            # Try general query first
            try:
                articles = self.newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    from_param=from_date,
                    page_size=50
                )['articles']
            except:
                # Fallback to top headlines if everything fails
                articles = self.newsapi.get_top_headlines(
                    q=query,
                    language='en',
                    page_size=50
                )['articles']

            return [{
                'title': a.get('title', 'No title'),
                'description': a.get('description', '')[:200] + '...' if a.get('description') else '',
                'published': a.get('publishedAt', ''),
                'url': a.get('url', '')
            } for a in articles]

        except Exception as e:
            print(f"❌ News fetch error: {str(e)}")
            return []


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
#     print("\nAAPL (10 years monthly):")
#     aapl_data = fetcher.get_market_data('AAPL', interval='monthly', lookback_years=10)
#     print(f"Retrieved {len(aapl_data)} monthly bars")
#     if not aapl_data.empty:
#         print(f"From {aapl_data.index[0].date()} to {aapl_data.index[-1].date()}")
#
#     print("\n=== Intraday Data Test ===")
#     print("SPY (15min intervals):")
#     spy_data = fetcher.get_market_data('SPY', interval='15min')
#     print(f"Retrieved {len(spy_data)} intraday bars")
#
#     print("\n=== News Test ===")
#     news = fetcher.get_news('stock market', lookback_days=3)
#     print(f"Retrieved {len(news)} news articles")
#     if news:
#         print(f"First article: {news[0]['title']}")