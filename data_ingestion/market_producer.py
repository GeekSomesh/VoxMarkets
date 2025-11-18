# data_ingestion/market_producer.py
import json
import logging
from datetime import datetime
from kafka import KafkaProducer
import yfinance as yf
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataProducer:
    """Fetch stock market data and send to Kafka"""
    
    def __init__(self, kafka_broker='localhost:9092'):
        """Initialize market data producer"""
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        
        logger.info("✅ Market Data Producer initialized")
    
    def fetch_daily_prices(self, tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']):
        """
        Fetch daily OHLCV data for tickers
        
        Args:
            tickers: List of stock symbols
        
        Returns:
            Number of data points sent
        """
        
        count = 0
        
        for ticker in tickers:
            try:
                logger.info(f"📊 Fetching market data for {ticker}...")
                
                # Fetch last 30 days of data
                data = yf.download(
                    ticker, 
                    period='30d', 
                    progress=False,
                    auto_adjust=False
                )
                
                # Send each day's data to Kafka
                for date, row in data.iterrows():
                    try:
                        # Convert pandas values properly
                        open_price = float(row['Open'])
                        high_price = float(row['High'])
                        low_price = float(row['Low'])
                        close_price = float(row['Close'])
                        volume = int(row['Volume'])
                        
                        market_event = {
                            'ticker': ticker,
                            'date': date.strftime('%Y-%m-%d'),
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume,
                            'ingested_at': datetime.now().isoformat(),
                        }
                        
                        self.producer.send('market-data', value=market_event)
                        count += 1
                        
                        # FIX: Log with converted values (not Series objects)
                        logger.info(f"✅ {ticker} {date.strftime('%Y-%m-%d')}: Close ${close_price:.2f}, Volume {volume:,}")
                    
                    except Exception as e:
                        logger.error(f"❌ Error processing row for {ticker}: {e}")
                        continue
            
            except Exception as e:
                logger.error(f"❌ Error fetching {ticker}: {e}")
                continue
        
        self.producer.flush()
        logger.info(f"\n✅ Total market data points sent: {count}")
        return count

# Main execution
if __name__ == "__main__":
    producer = MarketDataProducer()
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
    producer.fetch_daily_prices(tickers)
    
    print("\n✅ Market data fetching complete!")
