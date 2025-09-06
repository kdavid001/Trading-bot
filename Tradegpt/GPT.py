import openai
from openai import OpenAI
from openai._exceptions import RateLimitError
from config import CONFIG
from data_fetcher import fetch_data
from google import genai
from news_processor import NewsProcessor
from news_scraper import News_scraper

news_scraper = News_scraper()

openai_client = OpenAI(api_key=CONFIG.get('open_ai', '')) if CONFIG.get('open_ai', '') else None
gemini_client = genai.Client(api_key=CONFIG.get('gemini_api', '')) if CONFIG.get('gemini_api', '') else None
deepseek_client = OpenAI(api_key=CONFIG.get('deepseek_api'), base_url="https://api.deepseek.com") if CONFIG.get('deepseek_api', '') else None

#TODO: Create a private Virtual enviroment for each project specifically this.

# Fetch last 48 intervals (~12 hours)
data, symbol, trading_type = fetch_data()


def news_processor():
    news = news_scraper.get_latest_article(symbol, trading_type)
    score = news_scraper.process_sentiment(news, trading_type)
    print(f"sentiment score: {score}")
    return score


"""pivot = (high + low + close) / 3
support1 = 2 * pivot - high
resistance1 = 2 * pivot - low"""


def process_gbt():
    if data is None or any([openai_client,gemini_client,deepseek_client]) is None:
        print("No data or OpenAI client available.")
        return

    recent_data = data.tail(48)
    recent_data_str = recent_data.reset_index().to_string(index=False)

    strategy = f"""
                    Hybrid RSI + EMA + Sentiment Strategy:
                - Use EMA200 as a trend filter:
                   * If price > EMA10 → uptrend, only take Buy signals.
                   * If price < EMA10 → downtrend, only take Sell signals.
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
