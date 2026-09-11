Basic Market Data Columns

Column	Meaning
Date	The specific trading day (e.g., 2025-06-14).
Open	The price at which a stock or asset first traded when the market opened.
High	The highest price reached during the trading day.
Low	The lowest price reached during the trading day.
Close	The final price at which the asset traded when the market closed.
Volume	The total number of shares or contracts traded that day. Indicates interest or activity in the asset.


⸻

Technical Indicators

Column	Meaning
Returns	Daily percentage change in price, usually calculated as (Close - Open) / Open. Used to measure profitability.
Volatility	Measures how much the asset’s price moves. Higher values suggest larger price swings and higher risk. Commonly measured using standard deviation.
OBV (On-Balance Volume)	A volume-based indicator that adds volume on up days and subtracts on down days. Used to confirm trends.
CMF (Chaikin Money Flow)	Combines price and volume to measure buying/selling pressure over a period. Positive CMF indicates accumulation (buying), negative indicates distribution (selling).
RSI (Relative Strength Index)	Momentum oscillator that ranges from 0 to 100. RSI > 70 is overbought, < 30 is oversold. Used to predict reversals.
MACD (Moving Average Convergence Divergence)	Trend-following momentum indicator. It’s the difference between short-term and long-term moving averages. Positive MACD = bullish, negative = bearish.
ATR (Average True Range)	Measures volatility by averaging the range between high and low over a period. A higher ATR means greater price movement.
BB_Upper, BB_Middle, BB_Lower	Bollinger Bands: show volatility and price extremes. Upper and Lower are 2 standard deviations from the Middle (a moving average). Price touching bands may signal overbought/oversold conditions.
KST (Know Sure Thing)	A momentum oscillator based on the smoothed rate of change. When KST crosses above its signal line, it may indicate a buy signal.
Squeeze	Indicates a volatility “squeeze” based on Bollinger Bands and Keltner Channels. A squeeze suggests a potential breakout or breakdown is coming.


⸻

Targets and Sentiment

Column	Meaning
Vol_Target (Volatility Target)	Indicates whether the stock is within a desired range of volatility (e.g., for risk management).
Trend_Target	Indicates the current trend direction (e.g., 1 = bullish trend, 0 = no clear trend, -1 = bearish trend). Often used for signal generation.
Reversal_Target	Predicts if a reversal is likely. May be based on indicators like RSI, MACD crossovers, or candlestick patterns.
News_Sentiment	A sentiment score (e.g., -1 to +1) based on the tone of recent news headlines or articles. Positive sentiment may lead to bullish price action.
Symbol	The trading symbol/ticker of the asset (e.g., AAPL for Apple Inc.).
