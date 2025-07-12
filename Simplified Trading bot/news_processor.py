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

    def compute_signal(self, articles):
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

# if __name__ == "__main__":
#     News = NewsProcessor()
#     sample_articles = [dict(title="Canada Has Postponed Its Plans To Impose Counter Tariffs On U S Aluminum Imports",
#                             description='Canada is delaying its plans to slap retaliatory tariffs on U.S. steel and '
#                                         'aluminum after President Donald Trump sent a letter extending the deadline '
#                                         'for trade negotiations between the two North American neighbors — though he '
#                                         'also threatened to impose higher tariffs. Mark Carney’s government was '
#                                         'preparing to double its countertariffs on U.S. metals on July 21 — to 50 '
#                                         'percent from 25 — but Trump’s letter has moved the prime minister off that '
#                                         'target. Two senior government officials told POLITICO that Canada will not '
#                                         'further retaliate against U.S. steel and aluminum on July 21,')]
#     print(News.compute_signal(sample_articles))
