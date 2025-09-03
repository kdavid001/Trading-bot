import openai
from openai import OpenAI
from openai._exceptions import RateLimitError
from config import CONFIG
from data_fetcher import fetch_data
from google import genai
from news_processor import NewsProcessor
from news_scraper import News_scraper

news_scraper = News_scraper()

client = OpenAI(api_key=CONFIG.get('open_ai', '')) if CONFIG.get('open_ai', '') else None

# Fetch last 48 intervals (~12 hours)
data, symbol, trading_type = fetch_data()


def news_processor():
    news = news_scraper.get_latest_article(symbol, trading_type)
    score = news_scraper.process_sentiment(news, trading_type)
    print(f"sentiment score: {score}")
    return score

def process_gbt():
    if data is None or client is None:
        print("No data or OpenAI client available.")
        return

    recent_data = data.tail(48)
    recent_data_str = recent_data.reset_index().to_string(index=False)

    strategy = f"""
    Hybrid RSI + EMA + Sentiment Strategy:
- Use EMA200 as a trend filter:
   * If price > EMA200 → uptrend, only take Buy signals.
   * If price < EMA200 → downtrend, only take Sell signals.
- RSI rules:
   * RSI < 30 → Buy (only if trend = up).
   * RSI > 70 → Sell (only if trend = down).
- Sentiment adjustment (news_score:
this is the news sentiment score performed by taking the recent news on {symbol}, 
score = {news_processor()}):
   * If sentiment > 0.3 → bias Buy (even if RSI is neutral).
   * If sentiment < -0.3 → bias Sell (even if RSI is neutral).
   * If sentiment contradicts RSI/trend → reduce position size or skip trade.
    """

    prompt = f"""
You are an intraday stock market analyst AI.
Here is the 15-min data for {symbol} for the last 12 hours:
{recent_data_str}

Strategy:
{strategy}
Based on this, predict the next 4 intervals (1 hour) 
with recommended actions (Buy/Sell/Hold) and reasoning for each interval.
"""

    try:
        # Attempt higher-tier model first
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
    except RateLimitError:
        print("GPT-4O quota exceeded, retrying with gpt-3.5-turbo...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
        except RateLimitError:
            print("GPT-3.5-turbo quota exceeded. Falling back to Gemini...")
            process_gemini(prompt)
            return

    print(response.choices[0].message.content)


def process_gemini(prompt):
    gemini_client = genai.Client(api_key=CONFIG.get('gemini_api', '')) if CONFIG.get('gemini_api', '') else None
    if gemini_client is None:
        print("No Gemini API key provided.")
        return "No response from Gemini"

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        print(response.text)  # <-- This prints the actual completion
        return response.text
    except Exception as e:
        print("Gemini API error:", e)
        return "No response from Gemini"


if __name__ == '__main__':
    process_gbt()
