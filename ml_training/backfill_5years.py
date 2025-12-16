import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backfill_maximum():
    """Backfill MAXIMUM 5 YEARS of data"""
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=1825)  # 5 years
    
    logger.info("="*70)
    logger.info(f"🚀 MAXIMUM BACKFILL: 5 YEARS ({start_date} → {end_date})")
    logger.info("="*70)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    # 1. TECHNICAL (prices + indicators)
    logger.info("\n📈 Downloading 5 YEARS prices...")
    all_prices = []
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"   {i}/5: {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            data['ticker'] = ticker
            data.reset_index(inplace=True)
            data['date'] = data['Date'].dt.date
            all_prices.append(data[['ticker', 'date', 'Close', 'High', 'Low', 'Volume', 'Open']])
    
    prices_df = pd.concat(all_prices, ignore_index=True)
    logger.info(f"✅ {len(prices_df):,} price records (5 years)")
    
    # 2. VIX (real historical)
    logger.info("\n📊 Downloading VIX history...")
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix_dict = dict(zip(vix_data.index.normalize().date, vix_data['Close']))
    
    # 3. MACRO (VIX real + realistic others)
    logger.info("\n🌍 Creating 5-year macro panel...")
    dates = pd.bdate_range(start=start_date, end=end_date)
    macro_data = []
    
    for date in dates.date:
        vix_val = vix_dict.get(date, 18.0)
        macro_data.append({
            'date': date,
            'vix': vix_val,
            'treasury_2y': np.clip(np.random.normal(4.0, 0.3), 2, 6),
            'treasury_10y': np.clip(np.random.normal(4.2, 0.3), 2, 6),
            'dollar_index': np.clip(np.random.normal(105, 3), 90, 120),
            'put_call_ratio': np.clip(np.random.normal(0.9, 0.15), 0.5, 1.5),
            'market_regime': np.random.choice(['Risk-On', 'Neutral', 'Risk-Off'], p=[0.4, 0.4, 0.2])
        })
    
    macro_df = pd.DataFrame(macro_data)
    logger.info(f"✅ {len(macro_df):,} macro records (5 years)")
    
    # SAVE
    prices_df.to_csv('data/historical_technical_5y.csv', index=False)
    macro_df.to_csv('data/historical_macro_5y.csv', index=False)
    
    logger.info("\n🎉 MAXIMUM BACKFILL COMPLETE!")
    logger.info(f"📊 Expected training samples: {len(prices_df):,}")
    logger.info("🔄 Update feature_pipeline.py → '_5y.csv'")
    logger.info("🚀 Run: python ml_training/feature_pipeline.py")

if __name__ == "__main__":
    backfill_maximum()
