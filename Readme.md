## Things to put into consideration

Check the Sentiment analysis and see if it is actually being traineed on past news and not just recent ones</li>
To Improve the model you'd have to periodically train it every week at most</li>
Stop Looking at the problem as a regression task think of it as a classification oroblem to determine if it would increase or decrease over a certain period stop trying to predict the price

Dataset columns
	
* high: Highest price during the time period.
* open: Price of the asset at the beginning of the time period.
* low: Lowest price during the time period.
* close: Closing price at the end of the time period.
* returns: Usually calculated as percentage change from the previous close (log returns or simple returns).
* volatility: A measure of price fluctuations over a time window, often using standard deviation.</li>
* rsi (Relative Strength Index): Momentum indicator measuring the speed and change of price movements (typically overbought/oversold signals).
* macd (Moving Average Convergence Divergence): Trend-following momentum indicator based on EMAs (Exponential Moving Averages).
* atr (Average True Range): Indicator of market volatility.
* bb_upper, bb_middle, bb_lower (Bollinger Bands): These represent the upper, middle (SMA), and lower bands around the price to detect volatility and trend changes.


<h1> When to Consider a "Good" Model: </h1>
For trading systems to be profitable, you need:

### Trend Prediction:

- Validation accuracy > 55% consistently

- Precision/recall both > 53%

- Reversal Prediction:

- Recall > 25% (capture 1/4 of opportunities)

- Precision > 40% (2.5x better than random)

- Volatility Prediction:

- Validation loss < 0.05 (5% error)

- [x] Remove the news sentiment from the data prediction it should only  rely on the news after it has made its prediction. 
- [ ] Try using adding this: save the model with the lowes validation loss
```
if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), f'model_{Timestamp}_{Trading_type}_{Asset_name}.pt')
        valid_loss_min = valid_loss
```
