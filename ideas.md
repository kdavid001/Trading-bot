<h1> Things to put into consideration</h1>

<li>Check the Sentiment analysis and see if it is actually being traineed on past news and not just recent ones</li>
<li>To Improve the model you'd have to periodically train it every week at most</li>
<li>Stop Looking at the problem as a regression task think of it as a classification oroblem to determine if it would increase or decrease over a certain period stop trying to predict the price</li>

<h2>Dataset columns</h2>
	<li>**open**: Price of the asset at the beginning of the time period.
	<li>**high**: Highest price during the time period.
	<li>**low**: Lowest price during the time period.
	<li>**close**: Closing price at the end of the time period.
	<li>**returns**: Usually calculated as percentage change from the previous close (log returns or simple returns).
	<li>**volatility**: A measure of price fluctuations over a time window, often using standard deviation.</li>
    <li>**rsi** (Relative Strength Index): Momentum indicator measuring the speed and change of price movements (typically overbought/oversold signals).
	<li>**macd** (Moving Average Convergence Divergence): Trend-following momentum indicator based on EMAs (Exponential Moving Averages).
	<li>**atr** (Average True Range): Indicator of market volatility.
	<li>bb_upper, bb_middle, bb_lower (Bollinger Bands): These represent the upper, middle (SMA), and lower bands around the price to detect volatility and trend changes.

⸻

_🔍 Advanced/Composite Indicators_
<li>kst (Know Sure Thing): A momentum oscillator based on the smoothed rate-of-change for four different time frames.</li>
<li>squeeze: Often refers to “TTM Squeeze” — a condition when Bollinger Bands are inside Keltner Channels, indicating low volatility that may precede a breakout.
<li>news_sentiment: Sentiment score derived from news headlines or articles, typically via NLP models. Measures market sentiment (positive/negative/neutral).
<li>volume: Number of shares/contracts traded in a time period — a key indicator of market activity and strength of a price move.

<h1> When to Consider a "Good" Model: </h1>
For trading systems to be profitable, you need:

Trend Prediction:

Validation accuracy > 55% consistently

Precision/recall both > 53%

Reversal Prediction:

Recall > 25% (capture 1/4 of opportunities)

Precision > 40% (2.5x better than random)

Volatility Prediction:

Validation loss < 0.05 (5% error)



Remove the news sentiment from the data prediction it should only  rely on the news after it has made its prediction. 

