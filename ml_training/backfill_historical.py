import logging
import pandas as pd
import numpy as np
import psycopg2
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backfill_technical(start_date, end_date):
    """Backfill technical indicators for all tickers"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    logger.info(f"📈 Backfilling technical data: {start_date} to {end_date}")
    
    all_data = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data['ticker'] = ticker
        data.reset_index(inplace=True)
        data['date'] = data['Date'].dt.date
        all_data.append(data[['ticker', 'date', 'Close', 'High', 'Low', 'Volume']])
    
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ {len(df)} technical records backfilled")
    return df

def backfill_macro(start_date, end_date):
    """Backfill macro features"""
    logger.info(f"🌍 Backfilling macro data: {start_date} to {end_date}")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    macro_data = []
    
    for date in dates.date:
        # Mock macro data (replace with real APIs)
        macro_data.append({
            'date': date,
            'vix': np.random.normal(18, 3),  # Realistic VIX
            'treasury_2y': np.random.normal(4.0, 0.2),
            'treasury_10y': np.random.normal(4.2, 0.2),
            'dollar_index': np.random.normal(105, 2),
            'put_call_ratio': np.random.normal(0.9, 0.1),
            'market_regime': 'Neutral'
        })
    
    df = pd.DataFrame(macro_data)
    logger.info(f"✅ {len(df)} macro records backfilled")
    return df

if __name__ == "__main__":
    # Backfill 6 months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    
    logger.info("="*60)
    logger.info("🚀 HISTORICAL DATA BACKFILL (6 months)")
    logger.info("="*60)
    
    # 1. Technical data (real yfinance)
    technical_df = backfill_technical(start_date, end_date)
    
    # 2. Macro data (mock for now - replace with real APIs)
    macro_df = backfill_macro(start_date, end_date)
    
    # 3. Save for feature pipeline
    technical_df.to_csv('data/historical_technical.csv', index=False)
    macro_df.to_csv('data/historical_macro.csv', index=False)
    
    logger.info("\n✅ BACKFILL COMPLETE!")
    logger.info(f"📊 Total samples expected: ~600")
    logger.info("🔄 Run feature_pipeline.py to process!")
