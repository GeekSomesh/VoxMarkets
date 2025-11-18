# data_ingestion/kafka_to_postgres.py
import json
import logging
import psycopg2
from kafka import KafkaConsumer
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaToPostgresConsumer:
    """Read from Kafka topics and store in PostgreSQL"""
    
    def __init__(self, db_config, kafka_broker='localhost:9092'):
        """
        Initialize consumer
        
        Args:
            db_config: PostgreSQL connection details
            kafka_broker: Kafka broker address
        """
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            'news-ingestion',
            'market-data',
            bootstrap_servers=kafka_broker,
            group_id='postgres-consumer-group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
        )
        
        # Initialize PostgreSQL connection
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        
        logger.info("✅ Kafka to Postgres Consumer initialized")
    
    def create_tables(self):
        """Create tables if they don't exist"""
        
        logger.info("📋 Creating tables...")
        
        # Create news table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS bronze_news (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10),
                title TEXT,
                description TEXT,
                url VARCHAR(500) UNIQUE,
                published_at TIMESTAMP,
                source VARCHAR(100),
                author VARCHAR(200),
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create market data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS bronze_market_data (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10),
                date DATE,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                ingested_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, date)
            );
        """)
        
        # Create indexes for fast queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ticker ON bronze_news(ticker);
            CREATE INDEX IF NOT EXISTS idx_news_date ON bronze_news(published_at DESC);
            CREATE INDEX IF NOT EXISTS idx_market_ticker ON bronze_market_data(ticker);
            CREATE INDEX IF NOT EXISTS idx_market_date ON bronze_market_data(date DESC);
        """)
        
        self.conn.commit()
        logger.info("✅ Tables created")
    
    def consume_messages(self):
        """Read from Kafka and store in PostgreSQL"""
        
        logger.info("🔄 Starting consumer loop...")
        
        try:
            for message in self.consumer:
                topic = message.topic
                value = message.value
                
                try:
                    if topic == 'news-ingestion':
                        self._insert_news(value)
                    
                    elif topic == 'market-data':
                        self._insert_market_data(value)
                
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("⛔ Consumer stopped by user")
        
        finally:
            self.conn.close()
            logger.info("🔌 Database connection closed")
    
    def _insert_news(self, article):
        """Insert news article into database"""
        
        try:
            query = """
                INSERT INTO bronze_news 
                (ticker, title, description, url, published_at, source, author, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING;
            """
            
            self.cursor.execute(query, (
                article['ticker'],
                article['title'],
                article['description'],
                article['url'],
                article['published_at'],
                article['source'],
                article.get('author', 'Unknown'),
                article['ingested_at']
            ))
            
            self.conn.commit()
            logger.info(f"✅ Stored news: {article['ticker']} - {article['title'][:50]}...")
        
        except Exception as e:
            logger.error(f"❌ Error inserting news: {e}")
            self.conn.rollback()
    
    def _insert_market_data(self, market_event):
        """Insert market data into database"""
        
        try:
            query = """
                INSERT INTO bronze_market_data 
                (ticker, date, open, high, low, close, volume, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, date) DO NOTHING;
            """
            
            self.cursor.execute(query, (
                market_event['ticker'],
                market_event['date'],
                market_event['open'],
                market_event['high'],
                market_event['low'],
                market_event['close'],
                market_event['volume'],
                market_event['ingested_at']
            ))
            
            self.conn.commit()
            logger.info(f"✅ Stored market data: {market_event['ticker']} {market_event['date']}")
        
        except Exception as e:
            logger.error(f"❌ Error inserting market data: {e}")
            self.conn.rollback()

# Main execution
if __name__ == "__main__":
    # PostgreSQL connection config
    db_config = {
        'host': 'localhost',
        'database': 'sentiment_db',
        'user': 'admin',
        'password': 'admin123'
    }
    
    # Initialize consumer
    consumer = KafkaToPostgresConsumer(db_config)
    
    # Create tables
    consumer.create_tables()
    
    # Start consuming messages
    consumer.consume_messages()
