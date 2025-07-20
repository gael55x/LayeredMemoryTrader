import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def download_historical_data(symbol='BTC/USDT', time_frame='1m', days=30, output_file='historical_data.csv'):
    """
    Downloads historical OHLCV data from Binance using ccxt and saves it to a CSV file.
    
    :param symbol: The trading symbol (e.g., 'BTC/USDT').
    :param time_frame: The time frame for the candles (e.g., '1m', '5m', '1h').
    :param days: The number of days of data to fetch.
    :param output_file: The name of the output CSV file.
    """
    exchange = ccxt.binance()
    
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    all_candles = []
    
    print(f"Fetching {days} days of historical data for {symbol}...")

    while since < exchange.milliseconds():
        try:
            # fetch_ohlcv returns a list of lists: [timestamp, open, high, low, close, volume]
            candles = exchange.fetch_ohlcv(symbol, time_frame, since)
            
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1  # Move to the next millisecond
            print(f"Fetched {len(candles)} candles up to {exchange.iso8601(since)}")

        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying in 30 seconds...")
            time.sleep(30)
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}")
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