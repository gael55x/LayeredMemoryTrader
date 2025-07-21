import yfinance as yf
import pandas as pd
import os

def download_historical_data(ticker='AAPL', start_date='2020-01-01', end_date='2024-08-31', output_file='historical_data.csv'):
    """
    Downloads historical OHLCV data from Yahoo Finance and saves it to a CSV file.
    """
    print(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print(f"No data found for ticker {ticker}.")
        return

    # Convert timestamp to a column named 'time'
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    # Add a placeholder for news
    df['news'] = pd.NA

    df.to_csv(output_file, index=False)
    print(f"Historical data saved to {output_file}")

if __name__ == "__main__":
    download_historical_data() 