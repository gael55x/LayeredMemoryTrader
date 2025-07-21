import pandas as pd
import numpy as np
import os

# This is a placeholder for a real data download script.
# In a real application, you would use an API to fetch historical data.

def generate_synthetic_data(start_date='2020-01-01', end_date='2024-01-01', freq='H'):
    date_rng = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = {
        'time': date_rng,
        'open': np.random.uniform(100, 200, size=(len(date_rng))),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.uniform(1, 10, size=(len(date_rng)))
    }
    df = pd.DataFrame(data)
    df['high'] = df['open'] + np.random.uniform(0, 10, size=(len(date_rng)))
    df['low'] = df['open'] - np.random.uniform(0, 10, size=(len(date_rng)))
    df['close'] = df['open'] + np.random.uniform(-5, 5, size=(len(date_rng)))
    return df

def add_synthetic_news(df):
    news = []
    for i in range(1, len(df)):
        price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
        if price_change > 0.02:
            news.append("Major company reports record earnings, market rallies.")
        elif price_change < -0.02:
            news.append("New regulations spark fears of a market downturn.")
        else:
            news.append(np.nan)
    news.insert(0, np.nan)
    df['news'] = news
    return df

if __name__ == "__main__":
    # Create a dummy historical_data.csv if it doesn't exist
    if not os.path.exists('historical_data.csv'):
        print("Generating synthetic historical data...")
        df = generate_synthetic_data()
        df = add_synthetic_news(df)
        df.to_csv('historical_data.csv', index=False)
        print("Synthetic data saved to historical_data.csv")
    else:
        print("historical_data.csv already exists. Appending news column.")
        df = pd.read_csv('historical_data.csv')
        if 'news' not in df.columns:
            df = add_synthetic_news(df)
            df.to_csv('historical_data.csv', index=False)
            print("News column added to historical_data.csv")
        else:
            print("News column already exists in historical_data.csv") 