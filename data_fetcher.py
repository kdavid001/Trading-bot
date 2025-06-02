# data_fetcher.py
import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from twelvedata import TDClient
from newsapi import NewsApiClient
from config import CONFIG
import time
from datetime import datetime, timedelta


class DataFetcher:
    def __init__(self):
        self.av_fx = ForeignExchange(CONFIG['alpha_vantage'])
        self.av_crypto = CryptoCurrencies(CONFIG['alpha_vantage'])
        self.td = TDClient(apikey=CONFIG['twelvedata'])
        self.newsapi = NewsApiClient(api_key=CONFIG['newsapi'])
        self.last_api_call = time.time()
        self.min_call_interval = 15  # Seconds between API calls (rate limiting)

    def _rate_limit(self):
        """Enforce API rate limits"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_api_call = time.time()

    def get_market_data(self, symbol, interval='15min', lookback=1000):
        """Fetch OHLCV data with proper interval conversion"""
        self._rate_limit()

        # Convert interval to TwelveData format
        td_interval_map = {
            'daily': '1day',
            'weekly': '1week',
            'monthly': '1month',
            # Add other mappings as needed
        }
        td_interval = td_interval_map.get(interval.lower(), interval)

        try:
            if '/' in symbol:  # Forex or Crypto pair
                base, quote = symbol.split('/')

                # Try TwelveData first if interval is supported
                if td_interval in ['1min', '5min', '15min', '30min', '45min',
                                   '1h', '2h', '4h', '8h', '1day', '1week', '1month']:
                    try:
                        data = self.td.time_series(
                            symbol=symbol,
                            interval=td_interval,
                            outputsize=min(lookback, 5000)
                        ).as_pandas()
                        return self._clean_data(data)
                    except Exception as e:
                        print(f"Twelvedata failed: {e}. Falling back to Alpha Vantage")

                # Alpha Vantage fallback
                if quote == 'USD':
                    if interval in ['daily', 'weekly', 'monthly']:
                        data, _ = self.av_crypto.get_digital_currency_daily(
                            symbol=base,
                            market='USD'
                        )
                    else:
                        raise ValueError("Alpha Vantage only supports daily/weekly/monthly for crypto")
                else:
                    # Forex intraday
                    data, _ = self.av_fx.get_currency_exchange_intraday(
                        from_symbol=base,
                        to_symbol=quote,
                        interval=interval
                    )
            else:  # Stocks
                data = self.td.time_series(
                    symbol=symbol,
                    interval=td_interval,
                    outputsize=lookback
                ).as_pandas()

            return self._clean_data(data)


        except Exception as e:

            print(f"❌ Critical error fetching {symbol}: {str(e)}")

            return pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume'])  # Return empty but structured DataFrame

    def _clean_data(self, data):
        """Standardize data format across sources"""
        if data.empty:
            print("⚠️ Received empty data from API")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(data).sort_index()

        print(f"Raw columns received: {list(data.columns)}")

        # Handle different column naming conventions
        column_map = {
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

        # Convert to numeric and handle missing values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if len(df) == 0:
            print("⚠️ All rows dropped during cleaning!")

        return df.dropna()

    def get_news(self, query, lookback_days=3):
        """Fetch financial news articles with error handling"""
        self._rate_limit()

        try:
            from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            articles = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                from_param=from_date,
                page_size=min(50, 100)  # Stay within free tier limits
            )['articles']

            return [{
                'title': a.get('title', ''),
                'description': a.get('description', ''),
                'publishedAt': a.get('publishedAt', ''),
                'url': a.get('url', '')
            } for a in articles]

        except Exception as e:
            print(f"Error fetching news for {query}: {str(e)}")
            return []


# # Example usage
# if __name__ == "__main__":
#     fetcher = DataFetcher()
#
#     # Test different data sources
#     print("Testing BTC/USD (crypto):")
#     btc_data = fetcher.get_market_data('BTC/USD', interval='daily')
#     print(btc_data.tail())
#
#     print("\nTesting EUR/USD (forex):")
#     eur_data = fetcher.get_market_data('EUR/USD', interval='15min')
#     print(eur_data.tail())
#
#     print("\nTesting SPY (stock):")
#     spy_data = fetcher.get_market_data('SPY', interval='1h')
#     print(spy_data.tail())
#
#     print("\nTesting News API:")
#     news = fetcher.get_news('Bitcoin')
#     print(news[:2])  # Print first 2 articles