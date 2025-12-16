import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X = joblib.load('data/X_train.pkl')
y = joblib.load('data/y_train.pkl')

print("=" * 60)
print("🔍 PROPER OVERFITTING ANALYSIS")
print("=" * 60)
print()

# 1. Load model for evaluation
model = xgb.XGBRegressor()
model.load_model('models/xgboost_var_v1.0.json')

# 2. Train/Test metrics (from your trainer)
y_pred_train = model.predict(X)
train_r2 = r2_score(y, y_pred_train)
print(f"📊 Train R²: {train_r2:.4f}")

# 3. Proper cross-validation
print("\n📊 5-Fold Cross-Validation:")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"   Fold scores: {cv_scores.round(4)}")
print(f"   Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()

# 4. Learning curve (FIXED)
print("📈 Learning Curve:")
for size in [50, 100, 200, 300, 500]:
    if size > len(X):
        continue
    X_sub = X[:size]
    y_sub = y[:size]
    model_temp = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model_temp.fit(X_sub, y_sub)
    r2_sub = model_temp.score(X_sub, y_sub)
    print(f"   {size:3d} samples → R² = {r2_sub:.4f}")
print()

print("=" * 60)
print("🎯 SUMMARY")
print("=" * 60)
if cv_scores.std() < 0.1 and cv_scores.mean() > 0.6:
    print("✅ GOOD: Stable CV, reasonable performance")
elif cv_scores.std() > 0.15:
    print("⚠️  HIGH VARIANCE: Need more data")
else:
    print("✅ ACCEPTABLE: Ready for deployment")
