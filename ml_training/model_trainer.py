# ml_training/model_trainer.py
import logging
import pandas as pd
import numpy as np
import psycopg2
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train XGBoost model for VaR prediction"""
    
    def __init__(self, db_config):
        """Initialize model trainer"""
        
        logger.info("=" * 60)
        logger.info("🤖 XGBOOST MODEL TRAINER")
        logger.info("=" * 60)
        
        self.conn = psycopg2.connect(**db_config)
        self.model = None
        self.feature_cols = None
        
        logger.info("✅ Connected to PostgreSQL\n")
    
    def load_training_data(self):
        """Load prepared training data"""
        
        logger.info("📊 Loading training data...")
        
        X = joblib.load('data/X_train.pkl')
        y = joblib.load('data/y_train.pkl')
        feature_cols = joblib.load('data/feature_cols.pkl')
        
        logger.info(f"✅ Loaded training data")
        logger.info(f"   X shape: {X.shape}")
        logger.info(f"   y shape: {y.shape}")
        logger.info(f"   Features: {len(feature_cols)}\n")
        
        self.feature_cols = feature_cols
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train/test"""
        
        logger.info(f"📊 Splitting data (80/20)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"✅ Data split:")
        logger.info(f"   Train: {X_train.shape[0]} samples")
        logger.info(f"   Test: {X_test.shape[0]} samples\n")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        
        logger.info("🤖 Training XGBoost model...")
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
    'max_depth': 4,              # Reduce tree depth
    'learning_rate': 0.05,       # Slower learning
    'subsample': 0.7,            # Use 70% of samples per tree
    'colsample_bytree': 0.7,     # Use 70% of features per tree
    'colsample_bylevel': 0.7,    # Use 70% of features per level
    'reg_alpha': 1.0,            # L1 regularization
    'reg_lambda': 2.0,           # L2 regularization
    'min_child_weight': 5,       # Minimum samples in leaf
    'gamma': 1.0,                # Min loss reduction to split
    'n_estimators': 200,         # More trees, lighter each
    'random_state': 42,
    'verbosity': 0

        }
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(X_train, y_train)

        
        logger.info(f"✅ Model trained\n")
        
        return self.model
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate model performance"""
        
        logger.info("📊 Evaluating model...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Train metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        logger.info(f"\n📈 TRAINING METRICS:")
        logger.info(f"   RMSE: {train_rmse:.6f}")
        logger.info(f"   MAE: {train_mae:.6f}")
        logger.info(f"   R² Score: {train_r2:.4f}")
        
        logger.info(f"\n📈 TESTING METRICS:")
        logger.info(f"   RMSE: {test_rmse:.6f}")
        logger.info(f"   MAE: {test_mae:.6f}")
        logger.info(f"   R² Score: {test_r2:.4f}\n")
        
        return {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_test_pred': y_test_pred,
            'y_test': y_test
        }
    
    def get_feature_importance(self, model, top_n=15):
        """Get feature importance"""
    
        logger.info(f"🎯 Feature Importance (Top {top_n}):")
        logger.info("")
    
    # Get importance directly as dictionary
        importance_dict = model.get_booster().get_score(importance_type='weight')
    
        if not importance_dict:
            logger.info("   No feature importance available")
            logger.info("")
            return pd.DataFrame()
    
    # Build list of (feature_name, importance_score)
        importance_list = []
        for feature_key, score in importance_dict.items():
        # feature_key is like 'f0', 'f1', etc.
            try:
                idx = int(feature_key.replace('f', ''))
                if idx < len(self.feature_cols):
                    feature_name = self.feature_cols[idx]
                else:
                    feature_name = feature_key
            except:
                feature_name = feature_key
        
            importance_list.append({
                'Feature': feature_name,
                'Importance': score
            })
    
    # Convert to dataframe and sort
        importance_df = pd.DataFrame(importance_list).sort_values('Importance', ascending=False)
    
    # Display top features
        for rank, (idx, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
            feature_name = str(row['Feature'])
            importance_val = int(row['Importance'])
            logger.info(f"   {rank:2d}. {feature_name:30s}: {importance_val:6d}")
    
        logger.info("")
        return importance_df

    
    def save_model(self, model, version='v1.0'):
        """Save trained model"""
        
        logger.info(f"💾 Saving model version {version}...")
        
        model.save_model(f'models/xgboost_var_{version}.json')
        joblib.dump(self.feature_cols, f'models/feature_cols_{version}.pkl')
        
        logger.info(f"✅ Model saved to models/\n")
    
    def store_model_metadata(self, metrics, feature_importance, version='v1.0'):
        """Store model metadata in database"""
        
        logger.info(f"💾 Storing model metadata...")
        
        query = """
            INSERT INTO gold_model_metadata
            (model_name, model_version, training_date, train_accuracy, test_accuracy, rmse, mape, num_features, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_version) DO NOTHING;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (
                    'XGBoost_VAR_Predictor',
                    version,
                    datetime.now().date(),
                    float(metrics['train_r2']),
                    float(metrics['test_r2']),
                    float(metrics['test_rmse']),
                    float(metrics['test_mae']), 
                    int(len(self.feature_cols)),
                    'deployed'
                ))
            
            # Store feature importance
            importance_query = """
                INSERT INTO gold_feature_importance
                (model_version, feature_name, importance_score, rank)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """
            
            for rank, (idx, row) in enumerate(feature_importance.head(20).iterrows(), 1):
                with self.conn.cursor() as cur:
                    cur.execute(importance_query, (
                        version,
                        row['Feature'],
                        float(row['Importance']),
                        rank
                    ))
            
            self.conn.commit()
            logger.info(f"✅ Model metadata stored\n")
        
        except Exception as e:
            logger.error(f"❌ Error storing metadata: {e}\n")
            self.conn.rollback()
    
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
        trainer = ModelTrainer(db_config)
        
        # Load data
        X, y = trainer.load_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        # Train model
        logger.info(f"{'='*60}")
        logger.info(f"🚀 TRAINING PHASE")
        logger.info(f"{'='*60}\n")
        
        model = trainer.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Evaluate
        logger.info(f"{'='*60}")
        logger.info(f"📊 EVALUATION PHASE")
        logger.info(f"{'='*60}\n")
        
        metrics = trainer.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Feature importance
        logger.info(f"{'='*60}")
        logger.info(f"🎯 FEATURE IMPORTANCE")
        logger.info(f"{'='*60}\n")
        
        importance_df = trainer.get_feature_importance(model, top_n=15)
        
        # Save model
        logger.info(f"{'='*60}")
        logger.info(f"💾 MODEL PERSISTENCE")
        logger.info(f"{'='*60}\n")
        
        trainer.save_model(model, version='v1.0')
        trainer.store_model_metadata(metrics, importance_df, version='v1.0')
        
        # Summary
        logger.info(f"{'='*60}")
        logger.info(f"✅ MODEL TRAINING COMPLETE!")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"📈 Final Test R² Score: {metrics['test_r2']:.4f}")
        logger.info(f"📈 Final Test RMSE: {metrics['test_rmse']:.6f}\n")
        
        trainer.close()
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import sys
        sys.exit(1)
