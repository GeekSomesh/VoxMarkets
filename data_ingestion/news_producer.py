# data_ingestion/news_producer.py
import os
import json
import logging
import requests
from datetime import datetime, timedelta
from kafka import KafkaProducer
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsProducer:
    """Fetch financial news and send to Kafka"""
    
    def __init__(self, news_api_key, kafka_broker='localhost:9092'):
        """
        Initialize news producer
        
        Args:
            news_api_key: Your NewsAPI.org API key
            kafka_broker: Kafka broker address
        """
        
        self.news_api_key = news_api_key
        self.news_api_url = "https://newsapi.org/v2/everything"
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=3,
            acks='all'
        )
        
        logger.info("✅ News Producer initialized")
    
    def fetch_news(self, tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM'], hours=168):
        """
        Fetch financial news for given tickers
        Changed: Increased hours to 168 (1 week) for better results
        
        Args:
            tickers: List of stock symbols
            hours: How many hours back to fetch
        
        Returns:
            List of articles fetched
        """
        
        # Use correct date format YYYY-MM-DD
        from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime('%Y-%m-%d')
        to_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        all_articles = []
        
        for ticker in tickers:
            try:
                logger.info(f"📰 Fetching news for {ticker} from {from_date} to {to_date}...")
                
                # IMPROVED: Removed restrictive domains filter
                # This was blocking most results
                params = {
                    'q': ticker,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'from': from_date,
                    'to': to_date,
                    'pageSize': 50,  # Increased from 10
                    'apiKey': self.news_api_key
                }
                
                logger.info(f"🔍 API Request: {params}")
                
                response = requests.get(self.news_api_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # DEBUG: Show API response
                logger.info(f"📊 API Response: status={data.get('status')}, totalResults={data.get('totalResults')}")
                
                if data['status'] != 'ok':
                    logger.warning(f"⚠️ API issue for {ticker}: {data.get('message')}")
                    continue
                
                articles = data.get('articles', [])
                logger.info(f"📰 Found {len(articles)} articles for {ticker}")
                
                # Process each article
                for article in articles:
                    # Skip articles with missing data
                    if not article.get('title') or not article.get('url'):
                        continue
                    
                    enriched_article = {
                        'ticker': ticker,
                        'title': article['title'],
                        'description': article.get('description', '') or '',
                        'url': article['url'],
                        'published_at': article['publishedAt'],
                        'source': article['source']['name'],
                        'author': article.get('author', '') or 'Unknown',
                        'ingested_at': datetime.utcnow().isoformat(),
                    }
                    
                    # Send to Kafka
                    try:
                        self.producer.send(
                            'news-ingestion',
                            value=enriched_article
                        )
                        all_articles.append(enriched_article)
                        logger.info(f"✅ Sent: {ticker} - {article['title'][:60]}...")
                    
                    except Exception as e:
                        logger.error(f"❌ Error sending to Kafka: {e}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Network error for {ticker}: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ Unexpected error for {ticker}: {e}")
                continue
        
        # Ensure all messages are sent
        self.producer.flush()
        
        logger.info(f"\n✅ Total articles ingested: {len(all_articles)}")
        return all_articles
    
    def run_continuous(self, tickers, interval_hours=6):
        """
        Run news fetching every N hours continuously
        
        Args:
            tickers: List of stock symbols
            interval_hours: How often to fetch
        """
        
        logger.info(f"🔄 Starting continuous news fetching every {interval_hours} hours...")
        
        while True:
            try:
                self.fetch_news(tickers, hours=24)  # Always fetch last 24 hours
                logger.info(f"⏳ Waiting {interval_hours} hours before next fetch...")
                time.sleep(interval_hours * 3600)
            
            except KeyboardInterrupt:
                logger.info("⛔ News producer stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous loop: {e}")
                logger.info("🔄 Retrying in 30 seconds...")
                time.sleep(30)

# Main execution
if __name__ == "__main__":
    # Get API key from environment or prompt user
    news_api_key = os.getenv('NEWSAPI_KEY')
    
    if not news_api_key:
        news_api_key = input("❌ Enter your NewsAPI key (get it from https://newsapi.org): ")
    
    if not news_api_key or len(news_api_key) < 10:
        print("❌ Invalid API key!")
        exit(1)
    
    # Test API key validity
    print("🧪 Testing API key...")
    test_url = "https://newsapi.org/v2/everything"
    test_params = {
        'q': 'Apple',
        'apiKey': news_api_key,
        'pageSize': 1
    }
    try:
        test_response = requests.get(test_url, params=test_params, timeout=5)
        test_data = test_response.json()
        if test_data['status'] == 'ok':
            print(f"✅ API key is valid! Access level shows: {test_data.get('totalResults', '?')} results available")
        else:
            print(f"❌ API error: {test_data.get('message')}")
            exit(1)
    except Exception as e:
        print(f"❌ API connection error: {e}")
        exit(1)
    
    # Initialize producer
    producer = NewsProducer(news_api_key=news_api_key)
    
    # Fetch news - search last 7 days for better results
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
    producer.fetch_news(tickers, hours=168)  # 1 week
    
    print("\n✅ News fetching complete!")
