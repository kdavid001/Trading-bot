# A new approach to the trading architecture problem.
### - Instead of training GPT on raw time series data, you can treat it as a reasoning engine that simulates a strategy. You give it:
	1.	Market context (latest prices, trends, indicators, news).
	2.	Your strategy rules (e.g., “Buy if RSI < 30, sell if RSI > 70”).
	3.	Prediction task (e.g., “Will this strategy be profitable over the next week?”).