import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

def fetch_price_data(symbol='btcusd'):
    """
    Fetches the latest price data for a given symbol from the Gemini public API.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    url = f"https://api.gemini.com/v1/pubticker/{symbol}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        # Create a DataFrame from the response
        df = pd.DataFrame([data])
        df['datetime'] = pd.to_datetime(df['volume']['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Select and rename columns for clarity
        df = df[['last', 'volume']].rename(columns={'last': 'price'})
        df['price'] = df['price'].astype(float)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching price data: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    price_df = fetch_price_data()
    if price_df is not None:
        print(price_df) 