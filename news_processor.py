# news_processor.py
from transformers import pipeline
from textblob import TextBlob
import nltk

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

    def process_news_batch(self, articles):
        """Process list of news articles"""
        sentiment_scores = []
        for article in articles:
            text = f"{article['title']}. {article['description']}"
            sentiment_scores.append(self.analyze_sentiment(text))
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0