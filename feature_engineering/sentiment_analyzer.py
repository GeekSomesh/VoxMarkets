# feature_engineering/sentiment_analyzer.py
import logging
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    """
    Analyze financial news sentiment using FinBERT
    FinBERT is pre-trained on financial news and reports
    """
    
    def __init__(self, db_config):
        """
        Initialize FinBERT sentiment analyzer
        
        Args:
            db_config: PostgreSQL connection config
        """
        
        logger.info("=" * 60)
        logger.info("🧠 FINBERT SENTIMENT ANALYZER")
        logger.info("=" * 60)
        
        # Check GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"\n💻 Device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("   Using CPU (slower but works)")
        
        # Load FinBERT model
        logger.info("\n📦 Loading FinBERT model...")
        try:
            self.model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # negative, neutral, positive
            )
            
            # Move to GPU/CPU
            self.model.to(self.device)
            self.model.eval()  # Evaluation mode
            
            logger.info(f"✅ FinBERT model loaded successfully")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Database connection
        try:
            self.conn = psycopg2.connect(**db_config)
            logger.info("✅ Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def analyze_single_article(self, text):
        """
        Analyze sentiment of a single article
        
        Args:
            text: Article text (title + description)
        
        Returns:
            Dictionary with sentiment scores
        """
        
        try:
            # Truncate to 512 tokens max
            if len(text) > 500:
                text = text[:500]
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            probs_np = probs.detach().cpu().numpy()
            
            # Labels: 0=negative, 1=neutral, 2=positive
            sentiment_score = float(probs_np[2] - probs_np[0])  # positive - negative (-1 to +1)
            confidence = float(probs_np.max())
            
            return {
                'sentiment_score': sentiment_score,
                'positive_prob': float(probs_np[2]),
                'neutral_prob': float(probs_np[1]),
                'negative_prob': float(probs_np[0]),
                'confidence': confidence,
                'success': True
            }
        
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_unprocessed_articles(self, batch_size=32):
        """
        Process all articles that don't have sentiment scores yet
        
        Args:
            batch_size: Number of articles to process at once
        
        Returns:
            Number of articles processed
        """
        
        logger.info(f"\n📊 Fetching unprocessed articles...")
        
        # Get articles without sentiment scores
        query = """
            SELECT id, ticker, title, description, published_at
            FROM bronze_news
            WHERE id NOT IN (SELECT news_id FROM silver_news_sentiment)
            ORDER BY published_at DESC
            LIMIT %s;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (batch_size,))
            articles = cur.fetchall()
        
        if not articles:
            logger.info("✅ No new articles to process")
            return 0
        
        logger.info(f"📰 Found {len(articles)} unprocessed articles")
        logger.info(f"📈 Processing in batches of {batch_size}...\n")
        
        sentiments_to_insert = []
        processed_count = 0
        failed_count = 0
        
        for idx, (article_id, ticker, title, description, pub_date) in enumerate(articles, 1):
            try:
                # Combine title + description for richer context
                text = f"{title}. {description}" if description else title
                
                # Analyze sentiment
                result = self.analyze_single_article(text)
                
                if result['success']:
                    sentiments_to_insert.append((
                        article_id,
                        result['sentiment_score'],
                        result['positive_prob'],
                        result['neutral_prob'],
                        result['negative_prob'],
                        result['confidence']
                    ))
                    
                    # Log every 10 articles
                    if idx % 10 == 0:
                        logger.info(f"  [{idx}/{len(articles)}] ✅ {ticker}: {title[:50]}...")
                    
                    processed_count += 1
                else:
                    failed_count += 1
            
            except Exception as e:
                logger.error(f"  ❌ Error processing article {article_id}: {e}")
                failed_count += 1
                continue
        
        # Insert all sentiments to database
        if sentiments_to_insert:
            logger.info(f"\n💾 Storing {len(sentiments_to_insert)} sentiment scores...")
            self._store_sentiments_batch(sentiments_to_insert)
            logger.info("✅ Stored successfully")
        
        logger.info(f"\n📊 Processing Summary:")
        logger.info(f"  ✅ Processed: {processed_count}")
        logger.info(f"  ❌ Failed: {failed_count}")
        logger.info(f"  📊 Total: {len(articles)}")
        
        return processed_count
    
    def _store_sentiments_batch(self, sentiments):
        """
        Batch insert sentiment scores to database
        
        Args:
            sentiments: List of tuples (news_id, sentiment_score, ...)
        """
        
        query = """
            INSERT INTO silver_news_sentiment 
            (news_id, sentiment_score, positive_prob, neutral_prob, negative_prob, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (news_id) DO NOTHING;
        """
        
        try:
            with self.conn.cursor() as cur:
                execute_batch(cur, query, sentiments, page_size=100)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing sentiments: {e}")
            self.conn.rollback()
            raise
    
    def calculate_daily_sentiment(self, ticker, date_str):
        """
        Calculate aggregated daily sentiment for a ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            date_str: Date string (e.g., '2025-11-20')
        
        Returns:
            Dictionary with daily sentiment metrics
        """
        
        query = """
            SELECT 
                COUNT(*) as num_articles,
                AVG(sns.sentiment_score) as daily_sentiment,
                AVG(sns.confidence) as avg_confidence,
                SUM(CASE WHEN sns.sentiment_score > 0.2 THEN 1 ELSE 0 END) as bullish_count,
                SUM(CASE WHEN sns.sentiment_score < -0.2 THEN 1 ELSE 0 END) as bearish_count,
                SUM(CASE WHEN sns.sentiment_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) as neutral_count
            FROM silver_news_sentiment sns
            JOIN bronze_news bn ON sns.news_id = bn.id
            WHERE bn.ticker = %s 
            AND DATE(bn.published_at) = %s;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (ticker, date_str))
            result = cur.fetchone()
        
        if not result or result[0] == 0:
            return None
        
        num_articles, daily_sentiment, avg_conf, bullish, bearish, neutral = result
        
        return {
            'ticker': ticker,
            'date': date_str,
            'daily_sentiment': float(daily_sentiment),
            'num_articles': int(num_articles),
            'avg_confidence': float(avg_conf),
            'bullish_count': int(bullish or 0),
            'bearish_count': int(bearish or 0),
            'neutral_count': int(neutral or 0),
        }
    
    def get_sentiment_summary(self):
        """Get summary statistics of sentiment analysis"""
        
        query = """
            SELECT 
                COUNT(*) as total,
                AVG(sentiment_score) as avg_sentiment,
                MIN(sentiment_score) as min_sentiment,
                MAX(sentiment_score) as max_sentiment,
                AVG(confidence) as avg_confidence
            FROM silver_news_sentiment;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
        
        if not result or result[0] == 0:
            return None
        
        total, avg_sent, min_sent, max_sent, avg_conf = result
        
        return {
            'total_analyzed': int(total),
            'avg_sentiment': float(avg_sent),
            'min_sentiment': float(min_sent),
            'max_sentiment': float(max_sent),
            'avg_confidence': float(avg_conf),
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Main execution
if __name__ == "__main__":
    
    logger.info("\n🚀 Starting sentiment analysis pipeline...\n")
    
    # Database config
    db_config = {
        'host': 'localhost',
        'database': 'sentiment_db',
        'user': 'admin',
        'password': 'admin123',
        'port': 5432
    }
    
    try:
        # Initialize analyzer
        analyzer = FinBERTSentimentAnalyzer(db_config)
        
        # Process unprocessed articles (in batches of 50)
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING ARTICLES")
        logger.info("=" * 60)
        
        total_processed = 0
        iteration = 1
        
        while True:
            logger.info(f"\n🔄 Batch {iteration}:")
            processed = analyzer.process_unprocessed_articles(batch_size=50)
            total_processed += processed
            
            if processed < 50:
                break  # No more articles
            
            iteration += 1
            time.sleep(2)  # Small delay between batches
        
        # Get summary
        logger.info("\n" + "=" * 60)
        logger.info("SENTIMENT ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        summary = analyzer.get_sentiment_summary()
        if summary:
            logger.info(f"\n📊 Overall Statistics:")
            logger.info(f"   Total Articles Analyzed: {summary['total_analyzed']}")
            logger.info(f"   Average Sentiment: {summary['avg_sentiment']:+.3f}")
            logger.info(f"   Sentiment Range: {summary['min_sentiment']:+.3f} to {summary['max_sentiment']:+.3f}")
            logger.info(f"   Average Confidence: {summary['avg_confidence']:.1%}")
        
        # Test daily aggregation
        logger.info(f"\n📅 Sample Daily Sentiment (AAPL):")
        from datetime import date, timedelta
        test_date = (date.today() - timedelta(days=1)).isoformat()
        daily = analyzer.calculate_daily_sentiment('AAPL', test_date)
        
        if daily:
            logger.info(f"   Date: {daily['date']}")
            logger.info(f"   Sentiment: {daily['daily_sentiment']:+.3f}")
            logger.info(f"   Articles: {daily['num_articles']}")
            logger.info(f"   Bullish: {daily['bullish_count']}, Neutral: {daily['neutral_count']}, Bearish: {daily['bearish_count']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ SENTIMENT ANALYSIS COMPLETE!")
        logger.info("=" * 60)
        
        analyzer.close()
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        sys.exit(1)
