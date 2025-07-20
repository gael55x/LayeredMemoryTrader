from data.price_fetcher import fetch_price_data
from data.sentiment_fetcher import fetch_sentiment_data
import pandas as pd
import time

class DataManager:
    def __init__(self, config):
        self.config = config
        self.price_data = pd.DataFrame()
        self.sentiment_data = pd.DataFrame()

    def fetch_data(self):
        """
        Fetches both price and sentiment data and stores them.
        """
        price_df = fetch_price_data()
        if price_df is not None:
            self.price_data = pd.concat([self.price_data, price_df])

        sentiment_df = fetch_sentiment_data()
        if sentiment_df is not None:
            self.sentiment_data = pd.concat([self.sentiment_data, sentiment_df])

    def start(self):
        """
        Starts the data fetching loop.
        """
        while True:
            self.fetch_data()
            print("Price data:")
            print(self.price_data.tail())
            print("\nSentiment data:")
            print(self.sentiment_data.tail())
            time.sleep(self.config['polling_intervals']['price_data'])

if __name__ == '__main__':
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_manager = DataManager(config)
    data_manager.start() 