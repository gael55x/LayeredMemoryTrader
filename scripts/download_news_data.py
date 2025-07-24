import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.news_fetcher import fetch_historical_news
import pandas as pd
import yaml

def download_and_integrate_news(tickers=['AAPL', 'MSFT', 'GOOG'], 
                               start_date='2020-01-01', 
                               end_date='2024-08-31',
                               news_output_file='historical_news.csv',
                               integrated_output_file='historical_data_with_news.csv'):
    """
    Downloads historical news data and integrates it with existing price data.
    """
    print(f"Downloading news data for {tickers} from {start_date} to {end_date}...")
    
    # Fetch news data
    news_df = fetch_historical_news(tickers, start_date, end_date, news_output_file)
    
    if news_df.empty:
        print("No news data found. Creating placeholder news data...")
        # Create placeholder news data
        news_df = create_placeholder_news(tickers, start_date, end_date)
    
    # Load existing price data
    try:
        price_df = pd.read_csv('historical_data.csv', parse_dates=['time'])
        print(f"Loaded existing price data with {len(price_df)} records")
    except FileNotFoundError:
        print("historical_data.csv not found. Please run download_historical_data.py first.")
        return
    
    # Integrate news with price data
    integrated_df = integrate_news_with_price(price_df, news_df)
    
    # Save integrated data
    integrated_df.to_csv(integrated_output_file, index=False)
    print(f"Integrated data saved to {integrated_output_file}")
    print(f"Final dataset has {len(integrated_df)} records")
    
    return integrated_df

def create_placeholder_news(tickers, start_date, end_date):
    """
    Create placeholder news data when no real news is available
    """
    print("Creating placeholder news data...")
    
    # Generate dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    placeholder_news = []
    for ticker in tickers:
        for date in date_range:
            # Create some realistic placeholder news
            if date.weekday() < 5:  # Only weekdays
                placeholder_news.append({
                    'date': date,
                    'title': f'Market update for {ticker}',
                    'description': f'Regular market activity for {ticker}',
                    'content': f'Standard trading day for {ticker} with normal market conditions.',
                    'source': 'Market Data',
                    'url': '',
                    'ticker': ticker,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral'
                })
    
    return pd.DataFrame(placeholder_news)

def integrate_news_with_price(price_df, news_df):
    """
    Integrate news data with price data by matching dates
    """
    print("Integrating news with price data...")
    
    # Convert news date to date only (remove time)
    if not news_df.empty:
        news_df['date_only'] = news_df['date'].dt.date
    else:
        news_df['date_only'] = pd.Series(dtype='object')
    
    # Convert price time to date only
    price_df['date_only'] = price_df['time'].dt.date
    
    # Group news by date and ticker, create summaries
    if not news_df.empty:
        news_summaries = []
        for (date, ticker), group in news_df.groupby(['date_only', 'ticker']):
            # Create a summary of news for this date/ticker
            titles = group['title'].tolist()
            sentiments = group.get('sentiment_label', ['neutral'] * len(group)).tolist()
            
            summary = f"News: {'; '.join([f'{title} ({sent})' for title, sent in zip(titles[:3], sentiments[:3])])}"
            news_summaries.append({
                'date_only': date,
                'ticker': ticker,
                'news_summary': summary
            })
        
        news_summary_df = pd.DataFrame(news_summaries)
        
        # Merge with price data
        integrated_df = price_df.merge(
            news_summary_df, 
            on=['date_only', 'ticker'], 
            how='left'
        )
    else:
        # No news data, just add empty news column
        integrated_df = price_df.copy()
        integrated_df['news_summary'] = pd.NA
    
    # Clean up
    integrated_df = integrated_df.drop('date_only', axis=1)
    
    # Fill missing news with placeholder
    integrated_df['news_summary'] = integrated_df['news_summary'].fillna('No significant news')
    
    return integrated_df

def update_config_with_news():
    """
    Update the config.yaml to include news data path
    """
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Add news data path to config
        if 'backtest' not in config:
            config['backtest'] = {}
        
        config['backtest']['news_data_path'] = 'historical_data_with_news.csv'
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("Updated config.yaml with news data path")
        
    except Exception as e:
        print(f"Error updating config: {e}")

if __name__ == '__main__':
    # Load config to get tickers and date range
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Get tickers from config or use defaults
        tickers = config.get('tickers', ['AAPL', 'MSFT', 'GOOG'])
        
        # Get date range from config
        training_start = config['backtest']['training_period']['start']
        testing_end = config['backtest']['testing_period']['end']
        
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default values...")
        tickers = ['AAPL', 'MSFT', 'GOOG']
        training_start = '2020-01-01'
        testing_end = '2024-08-31'
    
    # Download and integrate news
    integrated_data = download_and_integrate_news(
        tickers=tickers,
        start_date=training_start,
        end_date=testing_end
    )
    
    # Update config
    update_config_with_news()
    
    print("\nNews data download and integration complete!")
    print("You can now use the integrated data with news in your trading system.") 