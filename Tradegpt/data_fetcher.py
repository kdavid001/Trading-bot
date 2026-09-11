import yfinance as yf
from ta.momentum import RSIIndicator
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
import pandas as pd
from news_processor import NewsProcessor

news_processor = NewsProcessor()


def format_forex_symbol(symbol: str) -> str:
    """
    Converts a forex pair like 'EUR/USD' or 'usd/jpy'
    to Yahoo Finance format 'EURUSD=X'.
    """
    symbol = symbol.strip().upper()
    if '/' in symbol:
        base, quote = symbol.split('/')
        return f"{base}{quote}=X"
    return symbol


def calculate_tp_sl(entry_price, direction, rr_ratio=2, sl_pct=0.2):
    # sl_pct = stop loss in percent (0.2% = 0.002)
    if direction == "Buy":
        sl = entry_price * (1 - sl_pct)
        tp = entry_price * (1 + sl_pct * rr_ratio)
    elif direction == "Sell":
        sl = entry_price * (1 + sl_pct)
        tp = entry_price * (1 - sl_pct * rr_ratio)
    else:
        tp, sl = None, None
    return tp, sl


def fetch_data(symbol):
    trading_type = "forex"
    ###########################################################
    # symbol = input('Enter a symbol (e.g., EUR/USD or AAPL): ')
    # if '/' in symbol:
    #     symbol = format_forex_symbol(symbol)
    ###########################################################

    if trading_type == "forex":
        symbol = format_forex_symbol(symbol)
    print(f'Fetching data for symbol: {symbol}')
    data = yf.download(tickers=symbol, interval="15m", period="5d")

    if data.empty:
        print(f"No data fetched for {symbol}.")
        return None, symbol

    close_prices = data['Close']
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.squeeze()

    # Indicators
    data['RSI'] = RSIIndicator(close=close_prices, window=14).rsi()
    data['EMA10'] = EMAIndicator(close=close_prices, window=10).ema_indicator()
    data['EMA200'] = EMAIndicator(close=close_prices, window=200).ema_indicator()
    macd = MACD(close=close_prices)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()

    data = data[['Open', 'High', 'Low', 'Close', 'RSI', 'EMA10', 'EMA200', 'MACD', 'MACD_signal']]
    print("Fetched successfully")
    data.to_csv('data.csv')
    return data, trading_type
