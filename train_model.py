"""
Delivery Time Prediction - ML Pipeline
Author: [Your Name]
Assignment: W2_A1 - Full Stack ML Deployment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  DELIVERY TIME PREDICTION - ML PIPELINE")
print("=" * 55)

df = pd.read_csv("data/deliveries.csv")
print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(3).to_string())

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[STEP 1] Feature Engineering...")

# Encode categorical variables
le_source = LabelEncoder()
le_dest   = LabelEncoder()
le_weather = LabelEncoder()
le_vehicle = LabelEncoder()

df['source_encoded']  = le_source.fit_transform(df['source_city'])
df['dest_encoded']    = le_dest.fit_transform(df['destination_city'])
df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'])
df['vehicle_encoded'] = le_vehicle.fit_transform(df['vehicle_type'])

# Weight-distance interaction feature
df['weight_x_distance'] = df['parcel_weight_kg'] * df['distance_km']

feature_cols = [
    'source_encoded', 'dest_encoded',
    'parcel_weight_kg', 'distance_km',
    'weather_encoded', 'vehicle_encoded',
    'weight_x_distance'
]

X = df[feature_cols]
y = df['actual_delivery_hours']

print(f"   Features used: {feature_cols}")
print(f"   Target: actual_delivery_hours")

# ─────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
print("\n[STEP 2] Splitting dataset (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
print("\n[STEP 3] Training models...")

models = {
    "Linear Regression"       : LinearRegression(),
    "Random Forest"           : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"       : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    results[name] = {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 3)}
    print(f"\n   [{name}]")
    print(f"     RMSE : {rmse:.3f} hrs")
    print(f"     MAE  : {mae:.3f}  hrs")
    print(f"     R²   : {r2:.3f}")

# ─────────────────────────────────────────────
# 5. SELECT BEST MODEL
# ─────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["rmse"])
best_model = models[best_name]
print(f"\n[STEP 4] Best model selected: {best_name}")
print(f"         RMSE={results[best_name]['rmse']}  MAE={results[best_name]['mae']}  R²={results[best_name]['r2']}")

# ─────────────────────────────────────────────
# 6. HANDLE POOR PREDICTIONS (Retraining Check)
# ─────────────────────────────────────────────
print("\n[STEP 5] Checking for poor predictions...")

best_preds = best_model.predict(X_test_scaled)
errors = np.abs(best_preds - y_test.values)
poor_threshold = 5.0   # hours
poor_mask = errors > poor_threshold

if poor_mask.sum() > 0:
    print(f"   [WARNING] {poor_mask.sum()} predictions have error > {poor_threshold} hrs")
    print("   [ACTION]  Retraining with tuned hyperparameters...")

    tuned_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    )
    tuned_model.fit(X_train_scaled, y_train)
    tuned_preds = tuned_model.predict(X_test_scaled)

    tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
    if tuned_rmse < results[best_name]["rmse"]:
        print(f"   [INFO] Tuned model is better! RMSE improved to {tuned_rmse:.3f}")
        best_model = tuned_model
        best_name  = "Gradient Boosting (Tuned)"
    else:
        print(f"   [INFO] Original model kept. RMSE: {results[best_name]['rmse']}")
else:
    print(f"   [OK] All predictions within acceptable range (< {poor_threshold} hrs error)")

# ─────────────────────────────────────────────
# 7. SAVE MODEL + ENCODERS
# ─────────────────────────────────────────────
print("\n[STEP 6] Saving model artifacts...")
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/delivery_model.pkl")
joblib.dump(scaler,     "model/scaler.pkl")
joblib.dump({
    "source"  : le_source,
    "dest"    : le_dest,
    "weather" : le_weather,
    "vehicle" : le_vehicle,
}, "model/encoders.pkl")

# Save metadata for Node.js API
metadata = {
    "model_name"    : best_name,
    "features"      : feature_cols,
    "source_cities" : list(le_source.classes_),
    "dest_cities"   : list(le_dest.classes_),
    "weather_types" : list(le_weather.classes_),
    "vehicle_types" : list(le_vehicle.classes_),
    "metrics"       : results[best_name] if best_name in results else {"note": "tuned"},
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("   Saved: model/delivery_model.pkl")
print("   Saved: model/scaler.pkl")
print("   Saved: model/encoders.pkl")
print("   Saved: model/metadata.json")

# ─────────────────────────────────────────────
# 8. SAMPLE PREDICTION TEST
# ─────────────────────────────────────────────
print("\n[STEP 7] Sample prediction test...")

def predict_delivery(source, dest, weight, distance, weather, vehicle):
    src_enc = le_source.transform([source])[0]
    dst_enc = le_dest.transform([dest])[0]
    wth_enc = le_weather.transform([weather])[0]
    veh_enc = le_vehicle.transform([vehicle])[0]
    w_x_d   = weight * distance

    features = [[src_enc, dst_enc, weight, distance, wth_enc, veh_enc, w_x_d]]
    scaled   = scaler.transform(features)
    hrs      = best_model.predict(scaled)[0]
    return round(hrs, 1)

sample = predict_delivery("Mumbai", "Pune", 2.5, 148, "Clear", "Bike")
print(f"   Mumbai -> Pune | 2.5kg | 148km | Clear | Bike")
print(f"   Predicted delivery time: {sample} hours")

print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print("=" * 55)
