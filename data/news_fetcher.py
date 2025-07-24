import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
from typing import List, Dict, Optional
import re

load_dotenv()

class NewsFetcher:
    """
    Fetches historical news data from multiple free sources.
    """
    
    def __init__(self):
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        
    def fetch_newsapi_historical(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical news from NewsAPI (free tier: 100 requests/day)
        """
        if not self.newsapi_key:
            print("NEWSAPI_KEY not found. Skipping NewsAPI fetch.")
            return pd.DataFrame()
            
        try:
            # Convert company ticker to company name for better search
            company_name = self._get_company_name(ticker)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{company_name}" OR "{ticker}"',
                'from': start_date,
                'to': end_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.newsapi_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok' and data['articles']:
                articles = []
                for article in data['articles']:
                    articles.append({
                        'date': pd.to_datetime(article['publishedAt']),
                        'title': article['title'],
                        'description': article['description'],
                        'content': article['content'],
                        'source': article['source']['name'],
                        'url': article['url']
                    })
                
                df = pd.DataFrame(articles)
                df['ticker'] = ticker
                return df
            else:
                print(f"No articles found for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return pd.DataFrame()
    
    def fetch_alpha_vantage_news(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch news from Alpha Vantage (free tier: 500 requests/day)
        """
        if not self.alpha_vantage_key:
            print("ALPHA_VANTAGE_KEY not found. Skipping Alpha Vantage fetch.")
            return pd.DataFrame()
            
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'feed' in data:
                articles = []
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                for article in data['feed']:
                    article_date = pd.to_datetime(article['time_published'])
                    if start_dt <= article_date <= end_dt:
                        articles.append({
                            'date': article_date,
                            'title': article['title'],
                            'description': article.get('summary', ''),
                            'content': article.get('summary', ''),
                            'source': article['source'],
                            'url': article['url'],
                            'sentiment_score': article.get('overall_sentiment_score', 0),
                            'sentiment_label': article.get('overall_sentiment_label', 'neutral')
                        })
                
                df = pd.DataFrame(articles)
                df['ticker'] = ticker
                return df
            else:
                print(f"No news found for {ticker} from Alpha Vantage")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching from Alpha Vantage: {e}")
            return pd.DataFrame()
    
    def fetch_yahoo_finance_news(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch news from Yahoo Finance using web scraping (no API key needed)
        """
        try:
            import yfinance as yf
            
            # Get company info to find news
            ticker_obj = yf.Ticker(ticker)
            
            # Note: yfinance doesn't provide historical news, but we can get recent news
            # For historical data, we'll need to implement web scraping
            print(f"Yahoo Finance news fetch not implemented for historical data yet")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def fetch_reddit_news(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch news from Reddit r/investing and r/stocks (free, no API key needed)
        """
        try:
            import praw
            
            # This would require Reddit API credentials
            # For now, return empty DataFrame
            print("Reddit news fetch requires Reddit API credentials")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching from Reddit: {e}")
            return pd.DataFrame()
    
    def fetch_combined_news(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch news from all available sources and combine them
        """
        print(f"Fetching news for {ticker} from {start_date} to {end_date}...")
        
        all_news = []
        
        # Try NewsAPI
        newsapi_df = self.fetch_newsapi_historical(ticker, start_date, end_date)
        if not newsapi_df.empty:
            all_news.append(newsapi_df)
            print(f"Found {len(newsapi_df)} articles from NewsAPI")
        
        # Try Alpha Vantage
        alpha_df = self.fetch_alpha_vantage_news(ticker, start_date, end_date)
        if not alpha_df.empty:
            all_news.append(alpha_df)
            print(f"Found {len(alpha_df)} articles from Alpha Vantage")
        
        # Combine all results
        if all_news:
            combined_df = pd.concat(all_news, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['title', 'date'])
            combined_df = combined_df.sort_values('date')
            print(f"Total unique articles found: {len(combined_df)}")
            return combined_df
        else:
            print("No news articles found from any source")
            return pd.DataFrame()
    
    def _get_company_name(self, ticker: str) -> str:
        """
        Convert ticker to company name for better news search
        """
        company_names = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft Corporation',
            'GOOG': 'Alphabet Inc',
            'GOOGL': 'Alphabet Inc',
            'AMZN': 'Amazon.com Inc',
            'TSLA': 'Tesla Inc',
            'META': 'Meta Platforms Inc',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc',
            'ADBE': 'Adobe Inc',
            'CRM': 'Salesforce Inc',
            'ORCL': 'Oracle Corporation',
            'INTC': 'Intel Corporation',
            'AMD': 'Advanced Micro Devices',
            'QCOM': 'Qualcomm Inc',
            'CSCO': 'Cisco Systems Inc',
            'IBM': 'International Business Machines',
            'V': 'Visa Inc',
            'JPM': 'JPMorgan Chase & Co',
            'BAC': 'Bank of America Corp'
        }
        return company_names.get(ticker, ticker)
    
    def create_news_summary(self, news_df: pd.DataFrame) -> str:
        """
        Create a summary of news for a given date
        """
        if news_df.empty:
            return "No significant news found."
        
        # Group by date and create summaries
        summaries = []
        for date, group in news_df.groupby(news_df['date'].dt.date):
            date_str = date.strftime('%Y-%m-%d')
            articles = []
            
            for _, article in group.iterrows():
                title = article['title'][:100] + "..." if len(article['title']) > 100 else article['title']
                sentiment = article.get('sentiment_label', 'neutral')
                articles.append(f"- {title} ({sentiment})")
            
            summary = f"News for {date_str}:\n" + "\n".join(articles[:5])  # Limit to 5 articles per day
            summaries.append(summary)
        
        return "\n\n".join(summaries)

def fetch_historical_news(tickers: List[str], start_date: str, end_date: str, output_file: str = 'historical_news.csv'):
    """
    Main function to fetch historical news for multiple tickers
    """
    fetcher = NewsFetcher()
    all_news = []
    
    for ticker in tickers:
        print(f"\nFetching news for {ticker}...")
        news_df = fetcher.fetch_combined_news(ticker, start_date, end_date)
        
        if not news_df.empty:
            all_news.append(news_df)
            print(f"Successfully fetched {len(news_df)} articles for {ticker}")
        else:
            print(f"No news found for {ticker}")
        
        # Rate limiting to avoid hitting API limits
        time.sleep(1)
    
    if all_news:
        combined_news = pd.concat(all_news, ignore_index=True)
        combined_news.to_csv(output_file, index=False)
        print(f"\nSaved {len(combined_news)} total articles to {output_file}")
        return combined_news
    else:
        print("No news data found for any ticker")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOG']
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    news_data = fetch_historical_news(tickers, start_date, end_date)
    print(f"\nFetched {len(news_data)} total news articles") 