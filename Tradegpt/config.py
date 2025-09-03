# config.py
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
    'twelvedata': os.getenv('TWELVEDATA_KEY'),
    'newsapi': os.getenv('NEWSAPI_KEY'),
    'alpaca_key': os.getenv('ALPACA_KEY_ID'),
    'alpaca_secret': os.getenv('ALPACA_SECRET_KEY'),
    'open_ai': os.getenv('OPENAI_KEY'),
    'gemini_api': os.getenv('GEMINI_API_KEY')
}