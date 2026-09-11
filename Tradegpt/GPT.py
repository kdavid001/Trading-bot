import openai
from openai import OpenAI
from openai._exceptions import RateLimitError
from config import CONFIG
from data_fetcher import fetch_data
from google import genai
from google.genai import types
from news_processor import NewsProcessor
from news_scraper import News_scraper

news_scraper = News_scraper()

openai_client = OpenAI(api_key=CONFIG.get('open_ai', '')) if CONFIG.get('open_ai', '') else None
gemini_client = genai.Client(api_key=CONFIG.get('gemini_api', '')) if CONFIG.get('gemini_api', '') else None
deepseek_client = OpenAI(api_key=CONFIG.get('deepseek_api'), base_url="https://api.deepseek.com") if CONFIG.get('deepseek_api', '') else None

#TODO: Create a private Virtual enviroment for each project specifically this.

# Fetch last 48 intervals (~12 hours)
symbol = "GBP/JPY"
data, trading_type = fetch_data(symbol)


def news_processor():
    news = news_scraper.get_latest_article(symbol, trading_type)
    score = news_scraper.process_sentiment(news, trading_type)
    print(f"sentiment score: {score}")
    return score


"""pivot = (high + low + close) / 3
support1 = 2 * pivot - high
resistance1 = 2 * pivot - low"""


def process_gbt():
    if data is None or any([openai_client, gemini_client, deepseek_client]) is None:
        print("No data or OpenAI client available.")
        return

    recent_data = data.tail(48)
    recent_data_str = recent_data.reset_index().to_string(index=False)

    strategy = f"""
                    Hybrid RSI + EMA + Sentiment Strategy:
               - Use EMA200 as a trend filter:
                    * If price > EMA200 → uptrend
                    * If price < EMA200 → downtrend
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
    You are an intraday forex market analyst AI.

    Here is the 15-minute OHLC data for {symbol} over the last 12 hours:
    {recent_data_str}

    Strategy:
    {strategy}

    Rules:
    - Follow the defined EMA trend strictly.
    - Use a minimum risk-reward ratio of 1:2.
    - Stop loss must be placed beyond the most recent swing high/low.
    - Use partial profit-taking:
       * TP1 at 1R
       * TP2 at 2R
       * TP3 at 3R
    - After TP1 is hit, move Stop Loss to breakeven.
    - After TP2 is hit, trail Stop Loss below recent structure.
    - If conditions are unclear, choose Hold.

    Task:
    Identify ONE high-probability intraday trade setup.

    Output format:

    Trade Setup:
    - Action (Buy / Sell / Hold):
    - Reasoning:
    - Entry Price:
    - Stop Loss:

    Take Profit Levels:
    - TP1:
    - TP2:
    - TP3:

    Trade Management (Next 4 Intervals):
    - Interval 1:
    - Interval 2:
    - Interval 3:
    - Interval 4:

    Do NOT provide financial advice disclaimers.
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # check out gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}]
        )
        print("✅ Using OpenAI GPT")
        return response.choices[0].message.content
    except Exception as e:
        print("❌ OpenAI failed:", str(e))

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an intraday stock market analyst AI."},
                {"role": "user", "content": prompt}
            ])
        return response.choices[0].message.content
    except Exception as e:
        print("❌ DeepSeek failed:", str(e))

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        print("✅ Using Gemini")
        if hasattr(response, "text"):
            return response.text
        elif hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "⚠️ Gemini returned an empty response"
    except Exception as e:
            print("❌ Gemini failed:", str(e))
            return f"⚠️ Gemini error: {e}"


if __name__ == '__main__':
    result = process_gbt()
    print(result)