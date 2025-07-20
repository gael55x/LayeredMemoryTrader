import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def download_historical_data(symbol='btcusd', time_frame='1m', days=30, output_file='historical_data.csv'):
    """
    Downloads historical data from the Gemini API and saves it to a CSV file.
    
    :param symbol: The trading symbol (e.g., 'btcusd').
    :param time_frame: The time frame for the candles (e.g., '1m', '5m', '1h').
    :param days: The number of days of data to fetch.
    :param output_file: The name of the output CSV file.
    """
    url = f"https://api.gemini.com/v2/candles/{symbol}/{time_frame}"
    
    start_date = datetime.now() - timedelta(days=days)
    all_candles = []

    # Gemini API has a limit of 1000 candles per request
    while start_date < datetime.now():
        print(f"Fetching data from: {start_date}")
        
        # Convert start_date to milliseconds
        since = int(start_date.timestamp() * 1000)
        
        params = {
            'since': since,
            'limit': 1000  # Max limit
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            candles = response.json()
            
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # The last candle's timestamp is the new start date
            last_timestamp_ms = candles[-1][0]
            start_date = datetime.fromtimestamp(last_timestamp_ms / 1000)
            
            # To avoid hitting rate limits
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical data: {e}")
            return

    if not all_candles:
        print("No data was downloaded.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Historical data saved to {output_file}")


if __name__ == '__main__':
    download_historical_data() 