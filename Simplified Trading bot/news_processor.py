# news_processor.py
import nltk
import numpy as np
from transformers import pipeline

nltk.download('punkt')


class NewsProcessor:
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )

    def analyze_sentiment(self, text):
        """Get sentiment score from financial news"""
        result = self.sentiment_pipeline(text[:512])[0]
        score = result['score'] * (1 if result['label'] == 'positive' else -1)
        return score

    def parse_numeric_value(self, value_str):
        """
        Converts a value like '1.4%', '2.2B', '-900K' to a float.
        """
        if not value_str:
            return 0.0
        value_str = value_str.replace('%', '').replace(',', '').strip().upper()
        multiplier = 1.0

        if value_str.endswith('K'):
            multiplier = 1_000
            value_str = value_str[:-1]
        elif value_str.endswith('M'):
            multiplier = 1_000_000
            value_str = value_str[:-1]
        elif value_str.endswith('B'):
            multiplier = 1_000_000_000
            value_str = value_str[:-1]

        try:
            return float(value_str) * multiplier
        except ValueError:
            return 0.0  # fallback if conversion fails

    def compute_signal(self, articles, trading_type):
        """
        Process list of news articles
        This method analyzes sentiment only from the title,
        then applies a weight based on the impact level.
        """
        impact_weights = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4,
            "none": 0.1
        }

        sentiment_scores = []
        if trading_type == 'crypto':
            for article in articles:
                try:
                    sentiment = self.analyze_sentiment(article['title'])
                except Exception as e:
                    sentiment = 0.0  # Neutral fallback
                    print(f"Sentiment analysis failed: {e}")
                weight = impact_weights.get(article.get('impact', 'none'), 0.1)
                weighted_score = sentiment * weight
                sentiment_scores.append(weighted_score)
                # print(sentiment_scores)
            return np.tanh(sum(sentiment_scores))
        elif trading_type == "forex":
            for article in articles:
                try:
                    forecast = self.parse_numeric_value(article.get("forecast", "0"))
                    previous = self.parse_numeric_value(article.get("previous", "0"))
                    actual = self.parse_numeric_value(article.get("actual", ""))
                    # print(f"forcast and previous for each article{forecast, previous, actual}")

                    # Sentiment logic
                    sentiment = 0
                    if forecast > previous:
                        sentiment = 0.5  # bullish expectation
                    elif forecast < previous:
                        sentiment = -0.5  # bearish expectation

                    if actual is not None:
                        if actual > forecast:
                            sentiment += 0.5  # bullish surprise
                        elif actual < forecast:
                            sentiment -= 0.5  # bearish surprise

                    impact_weights = {
                        "high": 1.0,
                        "medium": 0.7,
                        "low": 0.4,
                        "none": 0.1
                    }

                    weight = impact_weights.get(article.get('impact', 'none'), 0.1)
                    # print(f"weight for this impact: {weight}")
                    weighted_score = sentiment * weight
                    # print(f"Sentiment analysis impact weight: {weighted_score}")
                    sentiment_scores.append(weighted_score)


                except Exception as e:
                    print(f"Error computing forex sentiment: {e}")
                    continue
            # print(sentiment_scores)
            return np.tanh(sum(sentiment_scores))


if __name__ == "__main__":
    News = NewsProcessor()
    sample_articles = [{'date': 'Mon\nJul 14', 'symbol': 'JPY',
'forecast': '-1.4%', 'previous': '-9.1%', 'impact': 'low'}, {'date': 'Mon\nJul 14', 'symbol': 'JPY', 'forecast':
'0.5%', 'previous': '0.5%', 'impact': 'low'}, {'date': 'Mon\nJul 14', 'symbol': 'JPY', 'forecast': '0.1%',
'previous': '0.3%', 'impact': 'low'}]
    print(News.compute_signal(sample_articles, trading_type="forex"))
