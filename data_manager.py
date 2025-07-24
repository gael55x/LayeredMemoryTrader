from data.price_fetcher import fetch_price_data
from data.sentiment_fetcher import fetch_sentiment_data
import pandas as pd
import time
import yaml

class DataManager:
    def __init__(self, config, backtest_mode=None):
        self.config = config
        self.backtest_mode = backtest_mode
        self.all_historical_data = pd.DataFrame()
        self.tickers = []
        
        if self.backtest_mode:
            self._load_backtest_data()

    def _load_backtest_data(self):
        """Loads and prepares historical data for backtesting."""
        # Try to load data with news first, fallback to regular data
        news_data_path = self.config['backtest'].get('news_data_path', 'historical_data_with_news.csv')
        data_path = self.config['backtest']['full_data_path']
        
        try:
            print(f"Attempting to load data with news from {news_data_path}...")
            self.all_historical_data = pd.read_csv(
                news_data_path, 
                parse_dates=['time']
            )
            print(f"Successfully loaded data with news from {news_data_path}")
        except FileNotFoundError:
            print(f"News data not found at {news_data_path}, loading regular data from {data_path}...")
            self.all_historical_data = pd.read_csv(
                data_path, 
                parse_dates=['time']
            )
            # Add placeholder news column if not present
            if 'news_summary' not in self.all_historical_data.columns:
                self.all_historical_data['news_summary'] = 'No significant news'
        
        self.tickers = self.all_historical_data['ticker'].unique()
        
        if self.backtest_mode == 'train':
            start_date = pd.to_datetime(self.config['backtest']['training_period']['start'])
            end_date = pd.to_datetime(self.config['backtest']['training_period']['end'])
        elif self.backtest_mode == 'test':
            start_date = pd.to_datetime(self.config['backtest']['testing_period']['start'])
            end_date = pd.to_datetime(self.config['backtest']['testing_period']['end'])
        else:
            raise ValueError("Invalid backtest mode specified. Choose 'train' or 'test'.")

        self.all_historical_data = self.all_historical_data[
            (self.all_historical_data['time'] >= start_date) & 
            (self.all_historical_data['time'] <= end_date)
        ]
        self.backtest_iterator = self.all_historical_data.groupby('ticker')
        print(f"Historical data for {self.backtest_mode}ing loaded for tickers: {self.tickers}.")
        
        # Check if news data is available
        if 'news_summary' in self.all_historical_data.columns:
            news_count = self.all_historical_data['news_summary'].notna().sum()
            print(f"News data available: {news_count} records with news information")
        else:
            print("No news data available in the dataset")

    def get_data_for_ticker(self, ticker):
        """Returns the historical data for a specific ticker."""
        if self.backtest_mode:
            try:
                return self.backtest_iterator.get_group(ticker).set_index('time')
            except KeyError:
                return pd.DataFrame()
        else:
            # Live mode would fetch data for a specific ticker
            return pd.DataFrame()

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Example of running in backtest mode
    data_manager = DataManager(config, backtest_mode='train')
    
    # Get data for a specific ticker
    aapl_data = data_manager.get_data_for_ticker('AAPL')
    print("AAPL Data:")
    print(aapl_data.head()) 