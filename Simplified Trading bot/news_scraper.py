import time

from selenium import webdriver
from selenium.webdriver.common.by import By

from news_processor import NewsProcessor

news_processor = NewsProcessor()

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)

max_time = time.time() + 60 * 60 * 24 * 7


class News_scraper:
    def __init__(self):
        # self.driver = webdriver.Chrome(options=options)
        timeout = time.time() + 5
        five_min = time.time() + 60 * 5

    def get_latest_article(self, asset_symbol, trading_type):
        global news
        options = webdriver.ChromeOptions()
        # options.add_experimental_option("detach", True)
        driver = webdriver.Chrome(options=options)
        news = []
        if trading_type == 'crypto':
            driver.get(f"https://www.cryptocraft.com/news")
            # main_id = driver.find_element(By.ID, "body flexposts")
            news_items = driver.find_elements(By.CSS_SELECTOR, "li.flexposts__item")
            max_news = 10  # be careful with this it will explode if you keep increasing it, not sure at what point
            # but it will cause something called a numerical instability due to the tanh(x) function I am using to reduce it

            for item in news_items[:max_news]:
                try:
                    title_elem = item.find_element(By.CSS_SELECTOR, ".flexposts__story-title a")
                    title = title_elem.text
                    relative_url = title_elem.get_attribute("href")
                    url = "https://www.cryptocraft.com" + relative_url if relative_url.startswith("/") else relative_url

                    timestamp_elem = item.find_element(By.CSS_SELECTOR, ".flexposts__time")
                    timestamp = timestamp_elem.get_attribute("title")

                    source_elem = item.find_element(By.CSS_SELECTOR, "[data-source]")
                    source = source_elem.get_attribute("data-source")
                    try:
                        impact_class = item.find_element(By.CSS_SELECTOR, ".flexposts__storyimpact").get_attribute(
                            "class")
                        if "high" in impact_class:
                            impact = "high"
                        elif "medium" in impact_class:
                            impact = "medium"
                        else:
                            impact = "low"
                    except:
                        impact = "none"  # no impact assigned visually

                    article = {
                        "title": title,
                        "url": url,
                        "timestamp": timestamp,
                        "source": source,
                        "impact": impact
                    }

                    # print(article)
                    news.append(article)
                except Exception as e:
                    print("Error parsing article:", e)
            driver.quit()
        # only checks for a week
        elif trading_type == 'forex':
            driver.get(f"https://www.forexfactory.com/calendar")
            """
            Actual > Forecast
            → Often
            bullish
            for the currency
                Actual < Forecast
            → Often
            bearish
            for the currency
            """
            # If
            # forecast > previous → the
            # market
            # expects
            # improvement → possible
            # bullish
            # pressure.
            # If
            # forecast < previous → market
            # expects
            # decline → bearish
            # bias.
            # bullish - increase
            # bearish - decrease
            # print(asset_symbol)
            last_date = ""

            news_items = driver.find_elements(By.CSS_SELECTOR,"tr.calendar__row.calendar__row--single-event, tr."
                                                              "calendar__row.calendar__row--alt")

            for item in news_items:
                try:
                    # Try to extract the date from the row
                    try:
                        date_elem = item.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__date .date")
                        last_date = date_elem.text
                    except:
                        pass  # Keep using last_date if not found

                    symbol = item.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__currency").text.upper()

                    try:
                        impact_class = item.find_element(By.CSS_SELECTOR,
                                                         ".calendar__cell.calendar__impact").get_attribute("class")
                        if "high" in impact_class:
                            impact = "high"
                        elif "medium" in impact_class:
                            impact = "medium"
                        else:
                            impact = "low"
                    except:
                        impact = "none"

                    try:
                        forecast = item.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__forecast").text
                    except:
                        forecast = ""

                    try:
                        previous = item.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__previous").text
                    except:
                        previous = ""

                    try:
                        actual = item.find_element(By.CSS_SELECTOR, ".calendar__cell.calendar__actual").text
                    except:
                        actual = ""

                    article = {
                        "date": last_date,
                        "symbol": symbol,
                        "forecast": forecast,
                        "previous": previous,
                        "actual": actual,
                        "impact": impact
                    }

                    news.append(article)
                except Exception as e:
                    error_message = f"Error parsing article: {e}"
            # print(error_message)
            # filtered_news
            # print(asset_symbol)
            base = asset_symbol[:3]
            quote = asset_symbol[3:6]
            filtered_news = [item for item in news if item['symbol'] in (base, quote)]
            # for item in news:
            #     print(base, quote)
            #     print(item["symbol"])
            #     if item["symbol"] in (base, quote):
            #         print(item)

            news = filtered_news
            driver.quit()
        return news

    def process_sentiment(self, news, trading_type):
        result = news_processor.compute_signal(news, trading_type)
        return result


if __name__ == "__main__":
    fetcher = News_scraper()
    trading_type = "forex"
    # Get latest news for different assets
    assets = [
        # {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
        # {'symbol': 'ETH/USD', 'news_query': 'Ethereum'},
        {'symbol': 'USDJPY=X', 'news_query': 'Dollar Yen'},
        # {'symbol': 'SPY', 'news_query': 'S&P 500'}
    ]
    asset_name = assets[0]['news_query']
    asset_symbol = assets[0]['symbol']
    print(f"Fetching news for {asset_name}, symbol: {asset_symbol}")
    news = fetcher.get_latest_article(asset_symbol, trading_type)
