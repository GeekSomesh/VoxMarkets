# feature_engineering/macro_features.py
import logging
import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MacroFeaturesCollector:
    """Collect macroeconomic features for portfolio risk analysis"""
    
    def __init__(self, db_config):
        """
        Initialize macro features collector
        
        Args:
            db_config: PostgreSQL connection config
        """
        
        logger.info("=" * 60)
        logger.info("🌍 MACRO FEATURES COLLECTOR")
        logger.info("=" * 60)
        
        try:
            self.conn = psycopg2.connect(**db_config)
            logger.info("✅ Connected to PostgreSQL\n")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
        
        # API Keys
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            logger.warning("⚠️  FRED_API_KEY not set. Treasury yields will be skipped.")
    
    def fetch_vix(self, days=30):
        """
        Fetch VIX (Volatility Index) from yfinance
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with VIX data
        """
        
        logger.info("📊 Fetching VIX data...")
        
        try:
            # Fetch VIX data
            vix = yf.download('^VIX', period=f'{days}d', progress=False, auto_adjust=True)
            
            # Reset index to have date as column
            vix_df = vix[['Close']].reset_index()
            vix_df.columns = ['date', 'vix']
            vix_df['vix'] = vix_df['vix'].astype(float)
            
            logger.info(f"✅ Fetched {len(vix_df)} VIX records")
            logger.info(f"   VIX Range: {vix_df['vix'].min():.2f} - {vix_df['vix'].max():.2f}\n")
            
            return vix_df
        
        except Exception as e:
            logger.error(f"❌ Error fetching VIX: {e}\n")
            return None
    
    def fetch_treasury_yields(self):
        """
        Fetch Treasury yields from FRED API
        
        Returns:
            Dictionary with 2Y, 5Y, 10Y yields
        """
        
        logger.info("📊 Fetching Treasury yields...")
        
        if not self.fred_api_key:
            logger.warning("⚠️  FRED API key not set. Skipping Treasury yields.\n")
            return None
        
        try:
            # FRED Series IDs
            series_ids = {
                '2Y': 'DGS2',      # 2-Year Treasury
                '5Y': 'DGS5',      # 5-Year Treasury
                '10Y': 'DGS10'     # 10-Year Treasury
            }
            
            yields = {}
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            
            for name, series_id in series_ids.items():
                try:
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'sort_order': 'desc',
                        'limit': 10  # Get last 10 to find latest non-null
                    }
                    
                    response = requests.get(base_url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        observations = data.get('observations', [])
                        
                        # Find latest non-null value
                        latest_value = None
                        for obs in observations:
                            value_str = obs.get('value', '.')
                            if value_str != '.' and value_str is not None:
                                try:
                                    latest_value = float(value_str)
                                    break
                                except ValueError:
                                    continue
                        
                        if latest_value is not None:
                            yields[f'treasury_{name.lower()}'] = latest_value
                            logger.info(f"   ✅ {name} Yield: {latest_value:.2f}%")
                        else:
                            logger.warning(f"   ⚠️  {name}: No valid data found")
                            yields[f'treasury_{name.lower()}'] = None
                    
                    else:
                        logger.warning(f"   ⚠️  {name} HTTP {response.status_code}")
                        yields[f'treasury_{name.lower()}'] = None
                
                except requests.Timeout:
                    logger.warning(f"   ⚠️  {name}: Request timeout")
                    yields[f'treasury_{name.lower()}'] = None
                
                except Exception as e:
                    logger.warning(f"   ⚠️  {name}: {str(e)[:50]}")
                    yields[f'treasury_{name.lower()}'] = None
            
            if any(v is not None for v in yields.values()):
                logger.info("")
                return yields
            else:
                logger.info("")
                return None
        
        except Exception as e:
            logger.error(f"❌ Error: {str(e)[:100]}\n")
            return None
    
    def fetch_dollar_index(self):
        """
        Fetch Dollar Index using alternative method
        
        Returns:
            Dollar Index value
        """
        
        logger.info("📊 Fetching Dollar Index...")
        
        try:
            # Try UUP ETF (tracks dollar index inversely)
            data = yf.download('UUP', period='1d', progress=False, auto_adjust=True)
            
            if data is not None and len(data) > 0:
                latest_price = float(data['Close'].iloc[-1])
                
                # Rough conversion from UUP to DXY equivalent
                dxy_equivalent = (latest_price - 20) * 5 + 90
                
                logger.info(f"✅ Dollar Index: {dxy_equivalent:.2f}\n")
                return dxy_equivalent
            
            logger.warning("⚠️  Could not fetch Dollar Index\n")
            return None
        
        except Exception as e:
            logger.error(f"❌ Error: {e}\n")
            return None
    
    def fetch_put_call_ratio(self):
        """
        Fetch Put/Call ratio from SPY options
        
        Returns:
            Put/Call ratio
        """
        
        logger.info("📊 Fetching Put/Call Ratio...")
        
        try:
            spy = yf.Ticker('SPY')
            expirations = spy.options
            
            if expirations and len(expirations) > 0:
                exp_date = expirations[0]
                
                try:
                    opts = spy.option_chain(exp_date)
                    puts_volume = opts.puts['volume'].sum()
                    calls_volume = opts.calls['volume'].sum()
                    
                    if calls_volume > 0:
                        put_call_ratio = float(puts_volume / calls_volume)
                        logger.info(f"✅ Put/Call Ratio: {put_call_ratio:.2f}\n")
                        return put_call_ratio
                
                except Exception as e:
                    logger.warning(f"⚠️  Error: {e}\n")
                    return None
            
            logger.warning("⚠️  No options data available\n")
            return None
        
        except Exception as e:
            logger.error(f"❌ Error: {e}\n")
            return None
    
    def calculate_yield_curve_slope(self, treasury_2y, treasury_10y):
        """
        Calculate yield curve slope (10Y - 2Y)
        
        Args:
            treasury_2y: 2-year yield
            treasury_10y: 10-year yield
        
        Returns:
            Slope value
        """
        
        if treasury_2y is not None and treasury_10y is not None:
            slope = float(treasury_10y - treasury_2y)
            
            if slope < 0:
                logger.warning(f"   ⚠️  INVERTED YIELD CURVE: {slope:.2f}%")
            else:
                logger.info(f"   ✅ Yield Curve Slope: {slope:.2f}%")
            
            return slope
        
        logger.info(f"   ⚠️  Cannot calculate slope (missing data)")
        return None
    
    def determine_market_regime(self, vix, yield_slope, dollar_index):
        """
        Determine market regime based on macro indicators
        
        Args:
            vix: VIX level
            yield_slope: 10Y - 2Y
            dollar_index: DXY level
        
        Returns:
            Market regime string
        """
        
        regime = "Neutral"
        
        if vix is not None and vix > 25:
            regime = "Risk-Off"
        elif yield_slope is not None and yield_slope < -0.5:
            regime = "Risk-Off"
        elif vix is not None and vix < 15:
            regime = "Risk-On"
        
        logger.info(f"   📈 Market Regime: {regime}")
        
        return regime
    
    def collect_all_macro_features(self):
        """
        Collect all macro features for today
        
        Returns:
            Dictionary with all macro data
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🌍 COLLECTING MACRO FEATURES")
        logger.info(f"{'='*60}\n")
        
        macro_data = {
            'date': datetime.now().date().isoformat(),
        }
        
        # Fetch VIX
        vix_df = self.fetch_vix(days=1)
        if vix_df is not None and len(vix_df) > 0:
            macro_data['vix'] = float(vix_df['vix'].iloc[-1])
        else:
            macro_data['vix'] = None
        
        # Fetch Treasury yields
        treasury_data = self.fetch_treasury_yields()
        if treasury_data:
            macro_data.update(treasury_data)
        else:
            macro_data['treasury_2y'] = None
            macro_data['treasury_5y'] = None
            macro_data['treasury_10y'] = None
        
        # Fetch Dollar Index
        macro_data['dollar_index'] = self.fetch_dollar_index()
        
        # Fetch Put/Call ratio
        macro_data['put_call_ratio'] = self.fetch_put_call_ratio()
        
        # Calculate Yield Curve Slope
        macro_data['yield_curve_slope'] = self.calculate_yield_curve_slope(
            macro_data.get('treasury_2y'),
            macro_data.get('treasury_10y')
        )
        
        # Determine Market Regime
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 MARKET REGIME ANALYSIS")
        logger.info(f"{'='*60}\n")
        
        macro_data['market_regime'] = self.determine_market_regime(
            macro_data.get('vix'),
            macro_data.get('yield_curve_slope'),
            macro_data.get('dollar_index')
        )
        
        return macro_data
    
    def store_macro_features(self, macro_data):
        """
        Store macro features in database
        
        Args:
            macro_data: Dictionary with macro data
        
        Returns:
            True if successful
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"💾 STORING MACRO FEATURES")
        logger.info(f"{'='*60}\n")
        
        query = """
            INSERT INTO silver_macro_features 
            (date, vix, treasury_2y, treasury_5y, treasury_10y, 
             yield_curve_slope, dollar_index, put_call_ratio, market_regime)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                vix = EXCLUDED.vix,
                treasury_2y = EXCLUDED.treasury_2y,
                treasury_5y = EXCLUDED.treasury_5y,
                treasury_10y = EXCLUDED.treasury_10y,
                yield_curve_slope = EXCLUDED.yield_curve_slope,
                dollar_index = EXCLUDED.dollar_index,
                put_call_ratio = EXCLUDED.put_call_ratio,
                market_regime = EXCLUDED.market_regime;
        """
        
        try:
            # Convert to native Python types
            vix = float(macro_data['vix']) if macro_data.get('vix') is not None else None
            treasury_2y = float(macro_data['treasury_2y']) if macro_data.get('treasury_2y') is not None else None
            treasury_5y = float(macro_data['treasury_5y']) if macro_data.get('treasury_5y') is not None else None
            treasury_10y = float(macro_data['treasury_10y']) if macro_data.get('treasury_10y') is not None else None
            yield_slope = float(macro_data['yield_curve_slope']) if macro_data.get('yield_curve_slope') is not None else None
            dxy = float(macro_data['dollar_index']) if macro_data.get('dollar_index') is not None else None
            pc_ratio = float(macro_data['put_call_ratio']) if macro_data.get('put_call_ratio') is not None else None
            
            with self.conn.cursor() as cur:
                cur.execute(query, (
                    macro_data['date'],
                    vix,
                    treasury_2y,
                    treasury_5y,
                    treasury_10y,
                    yield_slope,
                    dxy,
                    pc_ratio,
                    macro_data.get('market_regime')
                ))
            self.conn.commit()
            
            logger.info(f"✅ Stored macro features for {macro_data['date']}\n")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error storing: {e}\n")
            self.conn.rollback()
            return False
    
    def get_latest_macro_features(self):
        """Get latest macro features from database"""
        
        query = """
            SELECT * FROM silver_macro_features
            ORDER BY date DESC
            LIMIT 5;
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error retrieving: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Main execution
if __name__ == "__main__":
    
    logger.info("\n🚀 Starting macro features pipeline...\n")
    
    db_config = {
        'host': 'localhost',
        'database': 'sentiment_db',
        'user': 'admin',
        'password': 'admin123',
        'port': 5432
    }
    
    try:
        collector = MacroFeaturesCollector(db_config)
        macro_data = collector.collect_all_macro_features()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 MACRO FEATURES SUMMARY")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"Date: {macro_data['date']}")
        logger.info(f"VIX: {macro_data.get('vix', 'N/A')}")
        logger.info(f"Treasury 2Y: {macro_data.get('treasury_2y', 'N/A')}")
        logger.info(f"Treasury 10Y: {macro_data.get('treasury_10y', 'N/A')}")
        logger.info(f"Yield Slope: {macro_data.get('yield_curve_slope', 'N/A')}")
        logger.info(f"Dollar Index: {macro_data.get('dollar_index', 'N/A')}")
        logger.info(f"Put/Call: {macro_data.get('put_call_ratio', 'N/A')}")
        logger.info(f"Regime: {macro_data.get('market_regime', 'N/A')}")
        
        success = collector.store_macro_features(macro_data)
        
        if success:
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 LATEST MACRO FEATURES")
            logger.info(f"{'='*60}\n")
            
            latest = collector.get_latest_macro_features()
            if latest is not None:
                print(latest.to_string(index=False))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ MACRO FEATURES COMPLETE!")
        logger.info(f"{'='*60}\n")
        
        collector.close()
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import sys
        sys.exit(1)