import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# ----------------------------
# 1. Fetch historical data
# ----------------------------
symbol = "EURUSD=X"
data = yf.download(tickers=symbol, interval="15m", period="3mo")
data.dropna(inplace=True)

# ----------------------------
# 2. Compute indicators
# ----------------------------
close_prices = data['Close']
data['RSI'] = RSIIndicator(close=close_prices, window=14).rsi()
data['EMA200'] = EMAIndicator(close=close_prices, window=200).ema_indicator()

# ----------------------------
# 3. Example News Sentiment
# ----------------------------
# In reality you’d plug in your NewsProcessor score for each timestamp
# Here we simulate with random scores for demonstration
import numpy as np
np.random.seed(42)
data['Sentiment'] = np.random.uniform(-1, 1, len(data))  # replace with real sentiment

# ----------------------------
# 4. Generate hybrid signals
# ----------------------------
def hybrid_signal(row):
    # Trend filter
    if row['Close'] > row['EMA200']:
        trend = "up"
    else:
        trend = "down"

    # RSI rules with trend filter
    if row['RSI'] < 30 and trend == "up":
        signal = "Buy"
    elif row['RSI'] > 70 and trend == "down":
        signal = "Sell"
    else:
        signal = "Hold"

    # News sentiment adjustment
    if row['Sentiment'] > 0.3 and signal == "Hold" and trend == "up":
        signal = "Buy"
    elif row['Sentiment'] < -0.3 and signal == "Hold" and trend == "down":
        signal = "Sell"

    return signal

data['Signal'] = data.apply(hybrid_signal, axis=1)

# ----------------------------
# 5. Simulate portfolio
# ----------------------------
initial_cash = 1000
cash = initial_cash
position = 0
portfolio_value = []

for i, row in data.iterrows():
    if row['Signal'] == 'Buy' and cash > 0:
        position = cash / row['Close']
        cash = 0
    elif row['Signal'] == 'Sell' and position > 0:
        cash = position * row['Close']
        position = 0
    total_value = cash + position * row['Close']
    portfolio_value.append(total_value)

data['Portfolio'] = portfolio_value

# ----------------------------
# 6. Plot results
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Portfolio'], label='Portfolio Value')
plt.plot(data.index, data['Close'], label='EURUSD Price', alpha=0.5)
plt.title(f"Hybrid RSI+EMA+Sentiment Strategy Backtest on {symbol}")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# ----------------------------
# 7. Performance metrics
# ----------------------------
returns = data['Portfolio'].pct_change().dropna()
sharpe_ratio = returns.mean() / returns.std() * (252**0.5)
total_return = (portfolio_value[-1] / initial_cash - 1) * 100
print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")