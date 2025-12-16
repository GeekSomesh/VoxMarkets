# feature_engineering/technical_indicators.py
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
from ta import momentum, volatility, trend
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechnicalIndicatorAnalyzer:
    """Compute technical indicators for stocks"""
    
    def __init__(self, db_config):
        """
        Initialize technical indicator analyzer
        
        Args:
            db_config: PostgreSQL connection config
        """
        
        logger.info("=" * 60)
        logger.info("📊 TECHNICAL INDICATORS ANALYZER")
        logger.info("=" * 60)
        
        try:
            self.conn = psycopg2.connect(**db_config)
            logger.info("✅ Connected to PostgreSQL\n")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def fetch_price_data(self, ticker):
        """
        Fetch price data for a ticker from bronze layer
        
        Args:
            ticker: Stock symbol
        
        Returns:
            DataFrame with OHLCV data
        """
        
        query = """
            SELECT date, open, high, low, close, volume
            FROM bronze_market_data
            WHERE ticker = %s
            ORDER BY date ASC;
        """
        
        try:
            df = pd.read_sql(query, self.conn, params=(ticker,))
            
            if len(df) == 0:
                logger.warning(f"⚠️  No price data found for {ticker}")
                return None
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            logger.info(f"📈 Fetched {len(df)} price records for {ticker}")
            logger.info(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}\n")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error fetching data for {ticker}: {e}")
            return None
    
    def compute_rsi(self, df, period=14):
        """
        RSI (Relative Strength Index)
        - Measures momentum: 0-100
        - Above 70: Overbought (potential sell signal)
        - Below 30: Oversold (potential buy signal)
        - Around 50: Neutral
        """
        
        try:
            rsi = momentum.rsi(df['close'], window=period)
            logger.info(f"   ✅ RSI-{period} computed")
            return rsi
        except Exception as e:
            logger.error(f"   ❌ Error computing RSI: {e}")
            return None
    
    def compute_macd(self, df):
        """
        MACD (Moving Average Convergence Divergence)
        - Trend following momentum indicator
        - MACD line: 12-day EMA - 26-day EMA
        - Signal line: 9-day EMA of MACD
        - Histogram: MACD - Signal
        
        Interpretation:
        - MACD > Signal: Bullish
        - MACD < Signal: Bearish
        - Histogram > 0: Positive momentum
        """
        
        try:
            macd_line = trend.macd(df['close'])
            macd_signal = trend.macd_signal(df['close'])
            macd_diff = trend.macd_diff(df['close'])
            
            logger.info(f"   ✅ MACD computed")
            
            return {
                'macd': macd_line,
                'macd_signal': macd_signal,
                'macd_diff': macd_diff
            }
        except Exception as e:
            logger.error(f"   ❌ Error computing MACD: {e}")
            return None
    
    def compute_bollinger_bands(self, df, period=20):
        """
        Bollinger Bands
        - SMA ± (2 × standard deviation)
        - Upper Band: Resistance
        - Lower Band: Support
        - BB Position: (Close - Lower) / (Upper - Lower)
        
        Interpretation:
        - Price at upper band: Overbought
        - Price at lower band: Oversold
        - Price outside bands: Extreme move
        """
        
        try:
            bb_high = volatility.bollinger_hband(df['close'], window=period)
            bb_mid = volatility.bollinger_mavg(df['close'], window=period)
            bb_low = volatility.bollinger_lband(df['close'], window=period)
            
            # Position within bands (0 = lower band, 1 = upper band)
            bb_position = (df['close'] - bb_low) / (bb_high - bb_low)
            
            logger.info(f"   ✅ Bollinger Bands-{period} computed")
            
            return {
                'bb_high': bb_high,
                'bb_mid': bb_mid,
                'bb_low': bb_low,
                'bb_position': bb_position
            }
        except Exception as e:
            logger.error(f"   ❌ Error computing Bollinger Bands: {e}")
            return None
    
    def compute_atr(self, df, period=14):
        """
        ATR (Average True Range)
        - Measures market volatility
        - High ATR: High volatility (high risk)
        - Low ATR: Low volatility (low risk)
        
        Used for:
        - Setting stop losses
        - Position sizing
        - Volatility-based trading
        """
        
        try:
            atr = volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
            logger.info(f"   ✅ ATR-{period} computed")
            return atr
        except Exception as e:
            logger.error(f"   ❌ Error computing ATR: {e}")
            return None
    
    def compute_volatility(self, df, period=20):
        """
        Historical Volatility (20-day rolling)
        - Standard deviation of returns
        - High volatility: High price swings (risk)
        - Low volatility: Stable prices
        
        Used for:
        - Risk assessment
        - Options pricing
        - Position sizing
        """
        
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=period).std()
            
            logger.info(f"   ✅ Volatility-{period}d computed")
            return volatility
        except Exception as e:
            logger.error(f"   ❌ Error computing volatility: {e}")
            return None
    
    def compute_returns(self, df):
        """
        Price returns
        - 1-day return: (Today - Yesterday) / Yesterday
        - 5-day return: (Today - 5 days ago) / 5 days ago
        
        Used for:
        - Performance tracking
        - Correlation analysis
        - Risk calculations
        """
        
        try:
            returns_1d = df['close'].pct_change(periods=1)
            returns_5d = df['close'].pct_change(periods=5)
            
            logger.info(f"   ✅ Returns computed")
            
            return {
                'returns_1d': returns_1d,
                'returns_5d': returns_5d
            }
        except Exception as e:
            logger.error(f"   ❌ Error computing returns: {e}")
            return None
    
    def compute_all_indicators(self, ticker):
        """
        Compute ALL technical indicators for a ticker
        
        Args:
            ticker: Stock symbol
        
        Returns:
            DataFrame with all indicators
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Computing indicators for {ticker}")
        logger.info(f"{'='*60}\n")
        
        # Fetch price data
        df = self.fetch_price_data(ticker)
        if df is None:
            return None
        
        logger.info(f"Computing technical indicators:\n")
        
        # Momentum
        df['rsi_14'] = self.compute_rsi(df, period=14)
        
        # Trend
        macd_dict = self.compute_macd(df)
        if macd_dict:
            df['macd'] = macd_dict['macd']
            df['macd_signal'] = macd_dict['macd_signal']
            df['macd_diff'] = macd_dict['macd_diff']
        
        # Volatility
        bb_dict = self.compute_bollinger_bands(df, period=20)
        if bb_dict:
            df['bb_high'] = bb_dict['bb_high']
            df['bb_mid'] = bb_dict['bb_mid']
            df['bb_low'] = bb_dict['bb_low']
            df['bb_position'] = bb_dict['bb_position']
        
        df['atr'] = self.compute_atr(df, period=14)
        df['volatility_20d'] = self.compute_volatility(df, period=20)
        
        # Price Action
        returns_dict = self.compute_returns(df)
        if returns_dict:
            df['returns_1d'] = returns_dict['returns_1d']
            df['returns_5d'] = returns_dict['returns_5d']
        
        logger.info(f"\n✅ All indicators computed for {ticker}\n")
        
        return df
    
    def store_indicators(self, ticker, df):
        """
        Store technical indicators in silver layer
        
        Args:
            ticker: Stock symbol
            df: DataFrame with indicators
        """
        
        if df is None or len(df) == 0:
            logger.warning(f"⚠️  No data to store for {ticker}")
            return 0
        
        logger.info(f"💾 Storing indicators for {ticker}...")
        
        indicators_to_insert = []
        
        for date, row in df.iterrows():
            try:
                indicators_to_insert.append((
                    ticker,
                    date.date(),
                    float(row['rsi_14']) if not pd.isna(row['rsi_14']) else None,
                    float(row['macd']) if not pd.isna(row['macd']) else None,
                    float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,
                    float(row['macd_diff']) if not pd.isna(row['macd_diff']) else None,
                    float(row['bb_high']) if not pd.isna(row['bb_high']) else None,
                    float(row['bb_mid']) if not pd.isna(row['bb_mid']) else None,
                    float(row['bb_low']) if not pd.isna(row['bb_low']) else None,
                    float(row['bb_position']) if not pd.isna(row['bb_position']) else None,
                    float(row['atr']) if not pd.isna(row['atr']) else None,
                    float(row['volatility_20d']) if not pd.isna(row['volatility_20d']) else None,
                    float(row['returns_1d']) if not pd.isna(row['returns_1d']) else None,
                    float(row['returns_5d']) if not pd.isna(row['returns_5d']) else None,
                ))
            except Exception as e:
                logger.error(f"Error preparing row for {ticker} {date}: {e}")
                continue
        
        if not indicators_to_insert:
            logger.warning(f"⚠️  No valid data to insert for {ticker}")
            return 0
        
        # Batch insert
        query = """
            INSERT INTO silver_technical_indicators 
            (ticker, date, rsi_14, macd, macd_signal, macd_diff, 
             bb_high, bb_mid, bb_low, bb_position, atr, volatility_20d, 
             returns_1d, returns_5d)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO NOTHING;
        """
        
        try:
            with self.conn.cursor() as cur:
                execute_batch(cur, query, indicators_to_insert, page_size=100)
            self.conn.commit()
            
            logger.info(f"✅ Stored {len(indicators_to_insert)} records for {ticker}\n")
            return len(indicators_to_insert)
        
        except Exception as e:
            logger.error(f"❌ Error storing data for {ticker}: {e}")
            self.conn.rollback()
            return 0
    
    def process_all_tickers(self, tickers):
        """
        Process all tickers
        
        Args:
            tickers: List of stock symbols
        
        Returns:
            Dictionary with statistics
        """
        
        stats = {
            'total_processed': 0,
            'total_stored': 0,
            'failed': []
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 PROCESSING ALL TICKERS")
        logger.info(f"{'='*60}\n")
        
        for ticker in tickers:
            try:
                # Compute indicators
                df = self.compute_all_indicators(ticker)
                
                if df is not None:
                    # Store in database
                    stored = self.store_indicators(ticker, df)
                    
                    stats['total_processed'] += 1
                    stats['total_stored'] += stored
                else:
                    stats['failed'].append(ticker)
            
            except Exception as e:
                logger.error(f"❌ Error processing {ticker}: {e}")
                stats['failed'].append(ticker)
        
        return stats
    
    def get_summary_statistics(self):
        """Get summary statistics of all indicators"""
        
        query = """
            SELECT 
                ticker,
                COUNT(*) as num_records,
                ROUND(AVG(rsi_14)::numeric, 2) as avg_rsi,
                ROUND(AVG(volatility_20d)::numeric, 4) as avg_vol,
                ROUND(AVG(atr)::numeric, 2) as avg_atr,
                ROUND(AVG(returns_1d)::numeric, 4) as avg_return
            FROM silver_technical_indicators
            GROUP BY ticker
            ORDER BY ticker;
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Main execution
if __name__ == "__main__":
    
    logger.info("\n🚀 Starting technical indicators pipeline...\n")
    
    db_config = {
        'host': 'localhost',
        'database': 'sentiment_db',
        'user': 'admin',
        'password': 'admin123',
        'port': 5432
    }
    
    try:
        # Initialize analyzer
        analyzer = TechnicalIndicatorAnalyzer(db_config)
        
        # Process all tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
        stats = analyzer.process_all_tickers(tickers)
        
        # Print summary
        logger.info(f"{'='*60}")
        logger.info(f"📊 PROCESSING SUMMARY")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"✅ Total processed: {stats['total_processed']}")
        logger.info(f"✅ Total stored: {stats['total_stored']}")
        
        if stats['failed']:
            logger.error(f"❌ Failed tickers: {', '.join(stats['failed'])}")
        
        # Get statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 INDICATOR STATISTICS")
        logger.info(f"{'='*60}\n")
        
        summary = analyzer.get_summary_statistics()
        if summary is not None:
            print(summary.to_string(index=False))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ TECHNICAL INDICATORS COMPLETE!")
        logger.info(f"{'='*60}\n")
        
        analyzer.close()
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import sys
        sys.exit(1)
