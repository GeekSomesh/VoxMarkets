# feature_engineering/options_gamma.py
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptionsGammaAnalyzer:
    """Analyze options market signals and gamma exposure"""
    
    def __init__(self, db_config):
        """Initialize options gamma analyzer"""
        
        logger.info("=" * 60)
        logger.info("📈 OPTIONS GAMMA ANALYZER")
        logger.info("=" * 60)
        
        try:
            self.conn = psycopg2.connect(**db_config)
            logger.info("✅ Connected to PostgreSQL\n")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def fetch_options_data(self, ticker):
        """
        Fetch options data for a ticker
        
        Args:
            ticker: Stock symbol
        
        Returns:
            Dictionary with options metrics
        """
        
        logger.info(f"📊 Fetching options data for {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get option chain
            expirations = stock.options
            
            if not expirations or len(expirations) == 0:
                logger.warning(f"⚠️  No options data for {ticker}")
                return None
            
            # Use first expiration (nearest term)
            exp_date = expirations[0]
            
            try:
                opts = stock.option_chain(exp_date)
                puts = opts.puts
                calls = opts.calls
            except Exception as e:
                logger.warning(f"⚠️  Error fetching options chain: {e}")
                return None
            
            # Calculate metrics
            puts_volume = int(puts['volume'].sum())
            calls_volume = int(calls['volume'].sum())
            
            if calls_volume == 0:
                put_call_ratio = 0
            else:
                put_call_ratio = float(puts_volume / calls_volume)
            
            # Average IV (Implied Volatility)
            avg_put_iv = float(puts['impliedVolatility'].mean()) if len(puts) > 0 else 0
            avg_call_iv = float(calls['impliedVolatility'].mean()) if len(calls) > 0 else 0
            
            # IV Skew
            put_skew = float(avg_put_iv - avg_call_iv)
            
            # Calculate Gamma
            gamma_exposure = self.calculate_gamma_exposure(puts, calls)
            
            # IV Rank
            iv_rank = self.calculate_iv_rank(avg_call_iv)
            
            logger.info(f"   ✅ {ticker}: Puts={puts_volume}, Calls={calls_volume}, P/C={put_call_ratio:.2f}")
            logger.info(f"   ✅ Gamma Exposure: {gamma_exposure:.2f}")
            logger.info(f"   ✅ Put Skew: {put_skew:.3f}\n")
            
            return {
                'puts_volume': puts_volume,
                'calls_volume': calls_volume,
                'put_call_ratio': put_call_ratio,
                'put_iv': avg_put_iv,
                'call_iv': avg_call_iv,
                'put_skew': put_skew,
                'gamma_exposure': gamma_exposure,
                'iv_rank': iv_rank,
                'exp_date': exp_date
            }
        
        except Exception as e:
            logger.error(f"❌ Error fetching options for {ticker}: {e}\n")
            return None
    
    def calculate_gamma_exposure(self, puts, calls):
        """
        Calculate dealer gamma exposure
        
        Gamma > 0: Market structure supportive (dips bought)
        Gamma < 0: Market structure is brake pedal (rallies sold)
        
        Args:
            puts: Put options data
            calls: Call options data
        
        Returns:
            Gamma exposure score
        """
        
        try:
            # Calculate gamma exposure (simplified)
            # Real implementation would use Greeks calculation
            
            # Use volume-weighted gamma proxy
            put_gamma = (puts['volume'] * puts['impliedVolatility']).sum()
            call_gamma = (calls['volume'] * calls['impliedVolatility']).sum()
            
            # Net gamma (negative = dealers sold calls, bullish setup)
            net_gamma = float(call_gamma - put_gamma)
            
            # Normalize by total volume
            total_vol = puts['volume'].sum() + calls['volume'].sum()
            if total_vol > 0:
                gamma_score = net_gamma / total_vol
            else:
                gamma_score = 0
            
            return gamma_score
        
        except Exception as e:
            logger.warning(f"⚠️  Error calculating gamma: {e}")
            return 0
    
    def calculate_iv_rank(self, current_iv):
        """
        Calculate IV Rank (0-100)
        
        Args:
            current_iv: Current implied volatility
        
        Returns:
            IV Rank (0-100)
        """
        
        # Simplified IV Rank
        # Real implementation would use 52-week IV range
        # Using proxy: IV rank based on typical range
        
        try:
            # Typical IV range for SPX/stocks: 8% to 40%
            min_iv = 0.08
            max_iv = 0.40
            
            if current_iv < min_iv:
                iv_rank = 0
            elif current_iv > max_iv:
                iv_rank = 100
            else:
                iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
            
            return float(iv_rank)
        
        except Exception as e:
            logger.warning(f"⚠️  Error calculating IV rank: {e}")
            return 50  # Default to neutral
    
    def determine_gamma_level(self, gamma_exposure):
        """
        Determine gamma exposure level
        
        Args:
            gamma_exposure: Gamma exposure score
        
        Returns:
            Gamma level string
        """
        
        if gamma_exposure > 0.5:
            return "Extreme Long"
        elif gamma_exposure > 0.1:
            return "Long"
        elif gamma_exposure < -0.5:
            return "Extreme Short"
        elif gamma_exposure < -0.1:
            return "Short"
        else:
            return "Neutral"
    
    def determine_gamma_signal(self, gamma_exposure, put_call_ratio):
        """
        Determine trading signal from gamma
        
        Args:
            gamma_exposure: Gamma exposure score
            put_call_ratio: Put/Call ratio
        
        Returns:
            Signal string
        """
        
        # If gamma is positive and put/call is low: Bullish (dips bought)
        if gamma_exposure > 0.1 and put_call_ratio < 0.9:
            return "Bullish"
        # If gamma is negative and put/call is high: Bearish (rallies sold)
        elif gamma_exposure < -0.1 and put_call_ratio > 1.1:
            return "Bearish"
        else:
            return "Neutral"
    
    def process_ticker(self, ticker):
        """
        Process options data for a ticker
        
        Args:
            ticker: Stock symbol
        
        Returns:
            Dictionary with all metrics
        """
        
        opts_data = self.fetch_options_data(ticker)
        
        if opts_data is None:
            return None
        
        gamma_level = self.determine_gamma_level(opts_data['gamma_exposure'])
        gamma_signal = self.determine_gamma_signal(
            opts_data['gamma_exposure'],
            opts_data['put_call_ratio']
        )
        
        return {
            'ticker': ticker,
            'date': datetime.now().date().isoformat(),
            'gamma_exposure': opts_data['gamma_exposure'],
            'gamma_level': gamma_level,
            'iv_rank': opts_data['iv_rank'],
            'iv_percentile': opts_data['iv_rank'],
            'puts_volume': opts_data['puts_volume'],
            'calls_volume': opts_data['calls_volume'],
            'put_call_ratio': opts_data['put_call_ratio'],
            'put_skew': opts_data['put_skew'],
            'gamma_signal': gamma_signal
        }
    
    def store_gamma_data(self, gamma_data):
        """
        Store gamma data in database
        
        Args:
            gamma_data: List of dictionaries with gamma data
        
        Returns:
            Number of records stored
        """
        
        if not gamma_data:
            logger.warning("⚠️  No data to store")
            return 0
        
        logger.info(f"\n💾 Storing options gamma data...")
        
        query = """
            INSERT INTO silver_options_gamma 
            (ticker, date, gamma_exposure, gamma_level, iv_rank, iv_percentile,
             puts_volume, calls_volume, put_call_ratio, put_skew, gamma_signal)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO UPDATE SET
                gamma_exposure = EXCLUDED.gamma_exposure,
                gamma_level = EXCLUDED.gamma_level,
                iv_rank = EXCLUDED.iv_rank,
                iv_percentile = EXCLUDED.iv_percentile,
                puts_volume = EXCLUDED.puts_volume,
                calls_volume = EXCLUDED.calls_volume,
                put_call_ratio = EXCLUDED.put_call_ratio,
                put_skew = EXCLUDED.put_skew,
                gamma_signal = EXCLUDED.gamma_signal;
        """
        
        try:
            records = []
            for data in gamma_data:
                records.append((
                    data['ticker'],
                    data['date'],
                    float(data['gamma_exposure']),
                    data['gamma_level'],
                    float(data['iv_rank']),
                    float(data['iv_percentile']),
                    int(data['puts_volume']),
                    int(data['calls_volume']),
                    float(data['put_call_ratio']),
                    float(data['put_skew']),
                    data['gamma_signal']
                ))
            
            with self.conn.cursor() as cur:
                execute_batch(cur, query, records, page_size=100)
            
            self.conn.commit()
            
            logger.info(f"✅ Stored {len(records)} gamma records\n")
            return len(records)
        
        except Exception as e:
            logger.error(f"❌ Error storing gamma data: {e}\n")
            self.conn.rollback()
            return 0
    
    def get_gamma_summary(self):
        """Get summary of gamma data"""
        
        query = """
            SELECT 
                ticker,
                gamma_level,
                ROUND(gamma_exposure::numeric, 3) as gamma_exp,
                ROUND(iv_rank::numeric, 1) as iv,
                ROUND(put_call_ratio::numeric, 2) as pc_ratio,
                gamma_signal
            FROM silver_options_gamma
            ORDER BY date DESC, ticker;
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error retrieving gamma summary: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Main execution
if __name__ == "__main__":
    
    logger.info("\n🚀 Starting options gamma pipeline...\n")
    
    db_config = {
        'host': 'localhost',
        'database': 'sentiment_db',
        'user': 'admin',
        'password': 'admin123',
        'port': 5432
    }
    
    try:
        analyzer = OptionsGammaAnalyzer(db_config)
        
        # Process all tickers
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 PROCESSING OPTIONS GAMMA")
        logger.info(f"{'='*60}\n")
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        gamma_data = []
        
        for ticker in tickers:
            data = analyzer.process_ticker(ticker)
            if data:
                gamma_data.append(data)
        
        # Store in database
        stored = analyzer.store_gamma_data(gamma_data)
        
        # Print summary
        logger.info(f"{'='*60}")
        logger.info(f"📊 GAMMA EXPOSURE SUMMARY")
        logger.info(f"{'='*60}\n")
        
        summary = analyzer.get_gamma_summary()
        if summary is not None and len(summary) > 0:
            print(summary.to_string(index=False))
        else:
            logger.info("⚠️  No gamma data available yet")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ OPTIONS GAMMA ANALYSIS COMPLETE!")
        logger.info(f"{'='*60}\n")
        
        analyzer.close()
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import sys
        sys.exit(1)
