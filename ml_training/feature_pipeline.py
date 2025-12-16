# ml_training/feature_pipeline.py
import logging
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Load and prepare features for ML model"""
    
    def __init__(self, db_config):
        """Initialize feature pipeline"""
        
        logger.info("=" * 60)
        logger.info("📊 FEATURE PIPELINE")
        logger.info("=" * 60)
        
        self.conn = psycopg2.connect(**db_config)
        self.scaler_sentiment = StandardScaler()
        self.scaler_technical = StandardScaler()
        self.scaler_macro = MinMaxScaler(feature_range=(0, 1))
        
        logger.info("✅ Connected to PostgreSQL\n")
    
    def load_sentiment_features(self):
        """Load sentiment features grouped by date and ticker"""
        
        logger.info("📈 Loading sentiment features...")
        
        query = """
            SELECT 
                bn.ticker,
                DATE(bn.published_at) as date,
                AVG(sns.sentiment_score) as sentiment_score,
                COUNT(*) as num_articles,
                AVG(sns.confidence) as sentiment_confidence,
                SUM(CASE WHEN sns.sentiment_score > 0 THEN 1 ELSE 0 END) as bullish_articles,
                SUM(CASE WHEN sns.sentiment_score < 0 THEN 1 ELSE 0 END) as bearish_articles
            FROM silver_news_sentiment sns
            JOIN bronze_news bn ON sns.news_id = bn.id
            GROUP BY bn.ticker, DATE(bn.published_at)
            ORDER BY bn.ticker, DATE(bn.published_at);
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df)} sentiment records\n")
        return df
    
    def load_technical_features(self):
        """Load technical indicator features"""
    
        logger.info("📈 Loading technical features...")
    
    # Check if historical data exists
        try:
            hist_df = pd.read_csv('data/historical_technical_5y.csv')
            hist_df['date'] = pd.to_datetime(hist_df['date']).dt.date
            logger.info(f"✅ Loaded {len(hist_df)} historical technical records")
        
        # Load existing DB data
            query = """
            SELECT 
                ticker,
                date,
                rsi_14,
                macd,
                macd_signal,
                macd_diff,
                bb_position,
                atr,
                volatility_20d,
                returns_1d,
                returns_5d
            FROM silver_technical_indicators
            ORDER BY ticker, date;
        """
            db_df = pd.read_sql(query, self.conn)
            db_df['date'] = pd.to_datetime(db_df['date']).dt.date
        
        # Combine historical + existing
            combined_df = pd.concat([hist_df, db_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['ticker', 'date'], inplace=True)
            combined_df.sort_values(['ticker', 'date'], inplace=True)
        
            logger.info(f"✅ Combined: {len(combined_df)} total technical records\n")
            return combined_df
        
        except FileNotFoundError:
            logger.warning("⚠️  No historical data found, using DB only")
        
            query = """
            SELECT 
                ticker,
                date,
                rsi_14,
                macd,
                macd_signal,
                macd_diff,
                bb_position,
                atr,
                volatility_20d,
                returns_1d,
                returns_5d
            FROM silver_technical_indicators
            ORDER BY ticker, date;
        """
            df = pd.read_sql(query, self.conn)
            logger.info(f"✅ Loaded {len(df)} technical records\n")
            return df
    
    def load_macro_features(self):
        """Load macro features"""
    
        logger.info("📈 Loading macro features...")
    
    # Check if historical data exists
        try:
            hist_df = pd.read_csv('data/historical_macro_5y.csv')
            hist_df['date'] = pd.to_datetime(hist_df['date']).dt.date
            logger.info(f"✅ Loaded {len(hist_df)} historical macro records")
        
        # Load existing DB data
            query = """
            SELECT 
                date,
                vix,
                treasury_2y,
                treasury_10y,
                yield_curve_slope,
                dollar_index,
                put_call_ratio,
                CASE 
                    WHEN market_regime = 'Risk-On' THEN 1.0
                    WHEN market_regime = 'Neutral' THEN 0.0
                    ELSE -1.0
                END as market_regime_encoded
            FROM silver_macro_features
            ORDER BY date DESC;
        """
            db_df = pd.read_sql(query, self.conn)
            db_df['date'] = pd.to_datetime(db_df['date']).dt.date
        
        # Combine historical + existing
            combined_df = pd.concat([hist_df, db_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['date'], inplace=True)
            combined_df.sort_values('date', inplace=True)
        
            logger.info(f"✅ Combined: {len(combined_df)} total macro records\n")
            return combined_df
        
        except FileNotFoundError:
            logger.warning("⚠️  No historical macro data found, using DB only")
        
            query = """
            SELECT 
                date,
                vix,
                treasury_2y,
                treasury_10y,
                yield_curve_slope,
                dollar_index,
                put_call_ratio,
                CASE 
                    WHEN market_regime = 'Risk-On' THEN 1.0
                    WHEN market_regime = 'Neutral' THEN 0.0
                    ELSE -1.0
                END as market_regime_encoded
            FROM silver_macro_features
            ORDER BY date DESC;
        """
            df = pd.read_sql(query, self.conn)
            logger.info(f"✅ Loaded {len(df)} macro records\n")
            return df
    
    def load_options_features(self):
        """Load options gamma features"""
        
        logger.info("📈 Loading options features...")
        
        query = """
            SELECT 
                ticker,
                date,
                gamma_exposure,
                iv_rank,
                put_call_ratio,
                put_skew,
                CASE 
                    WHEN gamma_signal = 'Bullish' THEN 1.0
                    WHEN gamma_signal = 'Neutral' THEN 0.0
                    ELSE -1.0
                END as gamma_signal_encoded
            FROM silver_options_gamma
            ORDER BY ticker, date;
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df)} options records\n")
        return df
    
    def create_combined_features(self):
        """Combine all features into single training dataset"""
        
        logger.info(f"\n{'='*60}")
        logger.info("🔗 COMBINING ALL FEATURES")
        logger.info(f"{'='*60}\n")
        
        # Load all features
        sentiment_df = self.load_sentiment_features()
        technical_df = self.load_technical_features()
        macro_df = self.load_macro_features()
        options_df = self.load_options_features()
        
        # Merge sentiment + technical (same ticker/date)
        merged = technical_df.merge(
            sentiment_df,
            on=['ticker', 'date'],
            how='left'
        )
        
        # Merge with macro (macro applies to all tickers)
        merged = merged.merge(
            macro_df,
            on=['date'],
            how='left'
        )
        
        # Merge with options (same ticker/date)
        merged = merged.merge(
            options_df,
            on=['ticker', 'date'],
            how='left'
        )
        
        logger.info(f"✅ Combined features shape: {merged.shape}")
        logger.info(f"   Columns: {len(merged.columns)}")
        logger.info(f"   Rows: {len(merged)}\n")
        
        return merged
    
    def handle_missing_values(self, df):
        """Handle missing values intelligently"""
    
        logger.info("🔧 Handling missing values...")
    
    # Fill forward then backward for time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            # If still NaN, fill with mean
                df[col] = df[col].fillna(df[col].mean())
    
    # ✅ ADD THIS BLOCK HERE (before return)
        df[numeric_cols] = df[numeric_cols].clip(lower=-10, upper=10)
        logger.info("✅ Extreme values capped [-10, 10]")
    
        logger.info(f"✅ Missing values handled\n")
        return df

    
    def create_target_variable(self, df):
        """Create proper VaR target (next-day realized volatility)"""
    
        logger.info("🎯 Creating proper VaR target (next-day volatility)...")
    
    # Calculate NEXT DAY realized volatility (no leakage!)
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Forward shift returns to get next-day realized vol
        df['next_day_return'] = df.groupby('ticker')['returns_1d'].shift(-1)
        df['realized_vol'] = df['next_day_return'].abs() * np.sqrt(252)  # Annualized
    
    # VaR = 1.645 * realized volatility (95% confidence)
        df['var_95'] = -1.645 * df['realized_vol']
    
    # Drop last row per ticker (no next-day data)
        df = df.dropna(subset=['var_95'])
    
        logger.info(f"✅ Proper VaR created: {len(df)} valid samples")
        logger.info(f"   VaR range: {df['var_95'].min():.4f} to {df['var_95'].max():.4f}")
    
        return df

    
    def prepare_training_data(self):
        """Prepare complete training dataset"""
        
        # Combine all features
        df = self.create_combined_features()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Select feature columns
        exclude_cols = ['ticker', 'date', 'var_95', 'actual_return', 'actual_volatility',
                'Close', 'High', 'Low', 'Volume', 'Open', 'Adj Close']  # Raw prices

        feature_cols = [col for col in df.columns 
                if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].select_dtypes(include=['number']).fillna(0)
        y = df['var_95'].fillna(0)

        logger.info(f"✅ Clean features: {len(feature_cols)} numeric columns")
        logger.info(f"   Feature types: {X.dtypes.value_counts().to_dict()}")

        
        logger.info(f"\n{'='*60}")
        logger.info("📊 TRAINING DATA SUMMARY")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Target variable (VaR): Mean={y.mean():.4f}, Std={y.std():.4f}")
        logger.info(f"Feature columns: {feature_cols[:5]}... and {len(feature_cols)-5} more\n")
        
        return X, y, df, feature_cols
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Main execution
if __name__ == "__main__":
    
    db_config = {
        'host': 'localhost',
        'database': 'sentiment_db',
        'user': 'admin',
        'password': 'admin123',
        'port': 5432
    }
    
    try:
        pipeline = FeaturePipeline(db_config)
        X, y, df, feature_cols = pipeline.prepare_training_data()
        
        logger.info(f"✅ Feature pipeline complete!")
        logger.info(f"   X shape: {X.shape}")
        logger.info(f"   y shape: {y.shape}")
        
        # Save for next step
        joblib.dump(X, 'data/X_train.pkl')
        joblib.dump(y, 'data/y_train.pkl')
        joblib.dump(feature_cols, 'data/feature_cols.pkl')
        
        logger.info(f"\n✅ Data saved to data/ directory\n")
        
        pipeline.close()
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import sys
        sys.exit(1)
