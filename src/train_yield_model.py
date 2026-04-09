# laat working code using old dataset  


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib


# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("../data/final_dataset.csv")
print("Original dataset size:", df.shape)


# -------------------------------
# Data Cleaning
# -------------------------------

df = df.dropna()
df = df.drop_duplicates()


print("\nMissing values:")
print(df.isnull().sum())



print("Before cleaning:", df.shape)

# ----------new-------------------
# Step 1 — Remove zero or negative Area
df = df[df["Area"] > 0]

# Step 2 — Remove zero or negative Production
df = df[df["Production"] > 0]

# Step 3 — Recalculate Yield cleanly
df["Yield"] = df["Production"] / df["Area"]

# Step 4 — Remove unrealistic yield values
# Sugarcane is highest realistic crop at ~80 tons/ha
# Keeping max at 100 to be safe
df = df[df["Yield"] <= 100]
df = df[df["Yield"] > 0]

print("After cleaning:", df.shape)
print("\nYield stats after cleaning:")
print(df["Yield"].describe())


# -------------------------------
# Log Transform Target Variable
# -------------------------------

df["Yield"] = np.log1p(df["Yield"])


# -------------------------------
# Encode Categorical Variables
# -------------------------------

le_state  = LabelEncoder()
le_crop   = LabelEncoder()
le_season = LabelEncoder()
le_Dist = LabelEncoder()

df["State"]  = le_state.fit_transform(df["State"])
df["Crop"]   = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])
df["District_Name"] = le_Dist.fit_transform(df["District_Name"])

# df = df.drop(columns=["District_Name"])


# -------------------------------
# Feature Selection
# -------------------------------

features = ["State","District_Name", "Crop", "Season", "Area", "Rainfall", "Temperature"]

X = df[features]
y = df["Yield"]


# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")


# -------------------------------
# Model Definition
# -------------------------------

model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)


# -------------------------------
# Model Training
# -------------------------------

print("\nTraining XGBoost...")
model.fit(X_train, y_train)
print("Training complete!")


# -------------------------------
# Evaluation — Train vs Test
# -------------------------------

train_preds = model.predict(X_train)
test_preds  = model.predict(X_test)

train_mae  = mean_absolute_error(y_train, train_preds)
train_mse  = mean_squared_error(y_train, train_preds)
train_rmse = np.sqrt(train_mse)
train_r2   = r2_score(y_train, train_preds)

test_mae  = mean_absolute_error(y_test, test_preds)
test_mse  = mean_squared_error(y_test, test_preds)
test_rmse = np.sqrt(test_mse)
test_r2   = r2_score(y_test, test_preds)

print(f"\n{'='*40}")
print(f"{'Metric':<12} {'Train':>10} {'Test':>10}")
print(f"{'-'*40}")
print(f"{'MAE':<12} {train_mae:>10.4f} {test_mae:>10.4f}")
print(f"{'MSE':<12} {train_mse:>10.4f} {test_mse:>10.4f}")
print(f"{'RMSE':<12} {train_rmse:>10.4f} {test_rmse:>10.4f}")
print(f"{'R2 Score':<12} {train_r2:>10.4f} {test_r2:>10.4f}")
print(f"{'='*40}")


# -------------------------------
# Overfitting Check
# -------------------------------

gap = train_r2 - test_r2
print(f"\n📊 R² Gap (Train - Test): {gap:.4f}")

if gap < 0.02:
    print("✅ No Overfitting — Model generalizes well")
elif gap < 0.05:
    print("⚠️  Slight Overfitting — Acceptable but monitor")
elif gap < 0.10:
    print("⚠️  Moderate Overfitting — Consider tuning")
else:
    print("❌ High Overfitting — Reduce max_depth or n_estimators")


# -------------------------------
# Cross Validation
# -------------------------------

print(f"\n{'='*40}")
print("K-Fold Cross Validation (k=5)")
print(f"{'='*40}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print(f"CV R² Scores : {np.round(cv_scores, 4)}")
print(f"Mean CV R²   : {cv_scores.mean():.4f}")
print(f"Std CV R²    : {cv_scores.std():.4f}")

if cv_scores.mean() > 0.80:
    print("✅ Cross Validation R² is strong")
elif cv_scores.mean() > 0.65:
    print("⚠️  Cross Validation R² is moderate")
else:
    print("❌ Cross Validation R² is weak")


# -------------------------------
# Dataset Info
# -------------------------------

print(f"\n{'='*40}")
print("Dataset Info")
print(f"{'='*40}")
print(f"Total unique crops  : {df['Crop'].nunique()}")
print(f"Total unique states : {df['State'].nunique()}")


# -------------------------------
# Save Model
# -------------------------------

print(f"\n{'='*40}")
print("Saving Model & Encoders")
print(f"{'='*40}")

joblib.dump(model,      "../models/yield_model.pkl")
joblib.dump(le_state,   "../models/state_encoder.pkl")
joblib.dump(le_crop,    "../models/crop_encoder.pkl")
joblib.dump(le_season,  "../models/season_encoder.pkl")
joblib.dump(le_Dist,  "../models/Dist_encoder.pkl")

print("✅ Model saved    → models/yield_model.pkl")
print("✅ State encoder  → models/state_encoder.pkl")
print("✅ Crop encoder   → models/crop_encoder.pkl")
print("✅ Season encoder → models/season_encoder.pkl")
print("✅ Dist encoder → models/Dist_encoder.pkl")
print("\n🚀 All done! Model is ready for deployment.")


































