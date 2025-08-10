

# TODO: CURRENTLY NOT IN USE


import requests
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from newsapi import NewsApiClient
from config import CONFIG
from news_processor import NewsProcessor

news_processor = NewsProcessor()


class AssetNewsFetcher:
    """Minimal news fetcher that returns only the latest article for an asset"""

    def __init__(self, api_key: str = None):
        """
        Initialize with optional API key.
        If no key provided, will look for NEWS_API_KEY in environment variables.
        """
        # self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.api_key = CONFIG.get('newsapi', '') if CONFIG.get('newsapi') else None
        if not self.api_key:
            raise ValueError("No API key provided. Set NEWS_API_KEY environment variable")

        self.base_url = "https://newsapi.org/v2/everything"

    def get_latest_article(self, asset_name):
        """
        Get the single most recent news article about the asset
        Returns:
            Dictionary with article details or None if no articles found
            Format: {
                'title': str,
                'description': str,
                'url': str,
                'published_at': str (ISO format),
                'source': str
            }
        """
        params = {
            'q': asset_name,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 1,  # Only get the most recent article
            'apiKey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])

            if articles:
                article = articles[0]
                return {
                    'title': article['title'],
                    'description': article['description'],
                }
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return None

    def process_sentiment(self, news, trading_type):
        result = news_processor.compute_signal(news, trading_type)
        return result


# if __name__ == "__main__":
#     # Initialize with your API key (or set NEWS_API_KEY environment variable)
#     fetcher = AssetNewsFetcher()
#
#     # Get latest news for different assets
#     assets = [
#         # {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
#         {'symbol': 'ETH/USD', 'news_query': 'Ethereum'},
#         # {'symbol': 'USDJPY=X', 'news_query': 'Dollar Yen'},
#         # {'symbol': 'SPY', 'news_query': 'S&P 500'}
#     ]
#     asset_name = assets[0]['news_query']
#     print(f"Fetching news for {asset_name}")
#     news = fetcher.get_latest_article(asset_name)
#     print(news)
#     result = fetcher.process_sentiment([news])
#
#     print(result)
