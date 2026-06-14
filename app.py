"""
Flask API - Delivery Time Prediction
Exposes the trained ML model as a REST endpoint
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# ── Load model artifacts ────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

model    = joblib.load(os.path.join(MODEL_DIR, "delivery_model.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

print("[INFO] Model loaded:", metadata["model_name"])

# ── Helper ──────────────────────────────────
def encode_and_predict(source, dest, weight, distance, weather, vehicle):
    try:
        src_enc = encoders["source"].transform([source])[0]
    except ValueError:
        return None, f"Unknown source city: '{source}'. Valid: {metadata['source_cities']}"

    try:
        dst_enc = encoders["dest"].transform([dest])[0]
    except ValueError:
        return None, f"Unknown destination city: '{dest}'. Valid: {metadata['dest_cities']}"

    try:
        wth_enc = encoders["weather"].transform([weather])[0]
    except ValueError:
        return None, f"Unknown weather: '{weather}'. Valid: {metadata['weather_types']}"

    try:
        veh_enc = encoders["vehicle"].transform([vehicle])[0]
    except ValueError:
        return None, f"Unknown vehicle type: '{vehicle}'. Valid: {metadata['vehicle_types']}"

    w_x_d    = weight * distance
    features = [[src_enc, dst_enc, weight, distance, wth_enc, veh_enc, w_x_d]]
    scaled   = scaler.transform(features)
    hours    = model.predict(scaled)[0]

    # Clamp to realistic range
    hours = max(1.0, round(float(hours), 1))
    return hours, None


# ── Routes ──────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service" : "Delivery Time Prediction API",
        "version" : "1.0",
        "model"   : metadata["model_name"],
        "metrics" : metadata.get("metrics", {}),
        "endpoints": {
            "POST /predict" : "Predict delivery time",
            "GET  /info"    : "Model info & valid values",
            "GET  /health"  : "Health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model"         : metadata["model_name"],
        "source_cities" : metadata["source_cities"],
        "dest_cities"   : metadata["dest_cities"],
        "weather_types" : metadata["weather_types"],
        "vehicle_types" : metadata["vehicle_types"],
        "features"      : metadata["features"],
        "metrics"       : metadata.get("metrics", {})
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    required = ["source_city", "destination_city", "parcel_weight_kg",
                "distance_km", "weather_condition", "vehicle_type"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        weight   = float(data["parcel_weight_kg"])
        distance = float(data["distance_km"])
    except (TypeError, ValueError):
        return jsonify({"error": "parcel_weight_kg and distance_km must be numbers"}), 400

    if weight <= 0 or distance <= 0:
        return jsonify({"error": "Weight and distance must be positive values"}), 400

    hours, err = encode_and_predict(
        source   = data["source_city"],
        dest     = data["destination_city"],
        weight   = weight,
        distance = distance,
        weather  = data["weather_condition"],
        vehicle  = data["vehicle_type"]
    )

    if err:
        return jsonify({"error": err, "prediction_failed": True}), 422

    return jsonify({
        "source_city"       : data["source_city"],
        "destination_city"  : data["destination_city"],
        "parcel_weight_kg"  : weight,
        "distance_km"       : distance,
        "weather_condition" : data["weather_condition"],
        "vehicle_type"      : data["vehicle_type"],
        "predicted_delivery_hours" : hours,
        "estimated_delivery_days"  : round(hours / 24, 1),
        "confidence"        : "high" if hours < 24 else "medium"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
