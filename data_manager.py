from data.price_fetcher import fetch_price_data
from data.sentiment_fetcher import fetch_sentiment_data
import pandas as pd
import time
import yaml

class DataManager:
    def __init__(self, config, backtest_file=None):
        self.config = config
        self.price_data = pd.DataFrame()
        self.sentiment_data = pd.DataFrame()
        self.backtest_file = backtest_file
        
        if self.backtest_file:
            self._load_backtest_data()

    def _load_backtest_data(self):
        """Loads and prepares historical data for backtesting."""
        print(f"Loading historical data from {self.backtest_file}...")
        self.historical_data = pd.read_csv(
            self.backtest_file, 
            index_col='time', 
            parse_dates=True
        )
        self.backtest_iterator = self.historical_data.iterrows()
        print("Historical data loaded.")

    def fetch_data(self):
        """
        Fetches data. In backtesting mode, it yields the next historical data point.
        In live mode, it fetches from the APIs.
        """
        if self.backtest_file:
            try:
                # Get the next row from the historical data
                timestamp, row = next(self.backtest_iterator)
                price_df = pd.DataFrame([row], index=[timestamp])
                self.price_data = pd.concat([self.price_data, price_df])
                return True
            except StopIteration:
                print("End of backtest data.")
                return False
        else:
            # Live mode fetching
            price_df = fetch_price_data()
            if price_df is not None:
                self.price_data = pd.concat([self.price_data, price_df])

            sentiment_df = fetch_sentiment_data()
            if sentiment_df is not None:
                self.sentiment_data = pd.concat([self.sentiment_data, sentiment_df])
            return True

    def start_live(self):
        """
        Starts the live data fetching loop.
        """
        if self.backtest_file:
            print("Cannot start live fetching in backtest mode.")
            return

        while True:
            self.fetch_data()
            print("Price data:")
            print(self.price_data.tail())
            print("\nSentiment data:")
            print(self.sentiment_data.tail())
            time.sleep(self.config['polling_intervals']['price_data'])

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Example of running in backtest mode
    data_manager = DataManager(config, backtest_file='historical_data.csv')
    
    # Simulate fetching a few data points
    for _ in range(5):
        if not data_manager.fetch_data():
            break
        print(data_manager.price_data.tail(1))

    # To run in live mode, you would instantiate without the backtest_file:
    # live_data_manager = DataManager(config)
    # live_data_manager.start_live() 