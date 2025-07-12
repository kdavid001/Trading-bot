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
        self.driver = webdriver.Chrome(options=options)
        timeout = time.time() + 5
        five_min = time.time() + 60 * 5

    def get_latest_article(self, asset_name, trading_type):
        global news
        if trading_type == 'crypto':
            options = webdriver.ChromeOptions()
            options.add_experimental_option("detach", True)
            driver = webdriver.Chrome(options=options)
            news = []
            driver.get(f"https://www.cryptocraft.com/news")
            # main_id = driver.find_element(By.ID, "body flexposts")
            news_items = driver.find_elements(By.CSS_SELECTOR, "li.flexposts__item")
            max_news = 10  # be careful with this it will explode if you keep increasing it not sure at what point but it will
            # cause something called a numerical instability due to the tanh function I am using to reduce it

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
        elif trading_type == 'forex':
            pass
        return news

    def process_sentiment(self, news):
        result = news_processor.compute_signal(news)
        return result

#
# if __name__ == "__main__":
#     # Initialize with your API key (or set NEWS_API_KEY environment variable)
#     fetcher = News_scraper()
#     trading_type = "crypto"
#     # Get latest news for different assets
#     assets = [
#         # {'symbol': 'BTC/USD', 'news_query': 'Bitcoin'},
#         {'symbol': 'ETH/USD', 'news_query': 'Ethereum'},
#         # {'symbol': 'USDJPY=X', 'news_query': 'Dollar Yen'},
#         # {'symbol': 'SPY', 'news_query': 'S&P 500'}
#     ]
#     asset_name = assets[0]['news_query']
#     print(f"Fetching news for {asset_name}")
#     news = fetcher.get_latest_article(asset_name, trading_type)
#     print(news)
