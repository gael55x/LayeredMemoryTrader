import yfinance as yf
import pandas as pd
import os

def download_historical_data(tickers=['AAPL', 'MSFT', 'GOOG'], start_date='2020-01-01', end_date='2024-08-31', output_file='historical_data.csv'):
    """
    Downloads historical OHLCV data from Yahoo Finance for multiple tickers and saves it to a single CSV file.
    """
    print(f"Downloading historical data for {tickers} from {start_date} to {end_date}...")
    
    # Download data for all tickers
    df = yf.download(tickers, start=start_date, end=end_date)
    
    if df.empty:
        print(f"No data found for tickers {tickers}.")
        return

    # Restructure the DataFrame
    df = df.stack(level=1).rename_axis(['time', 'ticker']).reset_index()
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # Add a placeholder for news
    df['news'] = pd.NA

    df.to_csv(output_file, index=False)
    print(f"Historical data saved to {output_file}")

if __name__ == "__main__":
    download_historical_data() 