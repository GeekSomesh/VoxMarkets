import joblib
import pandas as pd
import numpy as np

# Load latest data
X = joblib.load('data/X_train.pkl')
y = joblib.load('data/y_train.pkl')

print("=" * 60)
print("🔍 DATA DIAGNOSTIC")
print("=" * 60)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print()

print("📊 X SUMMARY:")
print(f"  NaN count: {X.isna().sum().sum()}")
print(f"  Inf count: {(np.isinf(X).sum()).sum()}")
print(f"  Min value: {X.min().min()}")
print(f"  Max value: {X.max().max()}")
print(f"  dtypes: {X.dtypes.value_counts()}")
print()

print("📊 y SUMMARY:")
print(f"  NaN count: {y.isna().sum()}")
print(f"  Inf count: {np.isinf(y).sum()}")
print(f"  Min: {y.min():.6f}")
print(f"  Max: {y.max():.6f}")
print(f"  Mean: {y.mean():.6f}")
print(f"  Std: {y.std():.6f}")
print()

print("🔥 TOP 10 COLUMNS BY VARIANCE:")
var_df = pd.DataFrame({'var': X.var()}).sort_values('var', ascending=False)
print(var_df.head(10))
print()

print("⚠️  SUSPICIOUS COLUMNS (extreme values):")
suspicious = X.std() > 100
print(X.columns[suspicious].tolist())
