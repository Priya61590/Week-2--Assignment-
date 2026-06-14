# Delivery Time Prediction — ML + REST API

A machine learning pipeline that predicts parcel delivery time (in hours) across Indian cities, deployed as a REST API using Flask (Python) and Node.js.

---

## Project Structure

```
delivery-prediction/
├── data/
│   └── deliveries.csv        # 40 Indian delivery records
├── model/
│   ├── delivery_model.pkl    # Trained ML model
│   ├── scaler.pkl            # Feature scaler
│   ├── encoders.pkl          # Label encoders
│   └── metadata.json         # Model metadata & valid inputs
├── api/
│   └── server.js             # Node.js REST service (port 3000)
├── train_model.py            # ML training pipeline
├── app.py                    # Flask ML API (port 5000)
└── README.md
```

---

## Dataset

**File:** `data/deliveries.csv`  
**Records:** 40 real-world inspired Indian delivery orders

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `source_city` | Pickup city (e.g., Mumbai, Delhi) |
| `destination_city` | Drop city |
| `parcel_weight_kg` | Weight in kilograms |
| `distance_km` | Route distance in km |
| `weather_condition` | Clear / Cloudy / Rainy |
| `vehicle_type` | Bike / Van / Truck |
| `actual_delivery_hours` | Ground truth delivery time |

Cities covered: Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Lucknow, Jaipur, Surat, and more.

---

## ML Model

### Approach

Three regression models were compared:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 1.535 | 1.038 | 0.923 |
| Random Forest | 2.221 | 1.736 | 0.838 |
| Gradient Boosting | 1.711 | 1.335 | 0.904 |

**Best model:** Linear Regression (R² = 0.923)

### Features Used

- Source city (label-encoded)
- Destination city (label-encoded)
- Parcel weight (kg)
- Distance (km)
- Weather condition (label-encoded)
- Vehicle type (label-encoded)
- Weight × Distance (interaction feature)

### Handling Poor Predictions

The pipeline automatically detects predictions with error > 5 hours and triggers a retraining step with tuned Gradient Boosting hyperparameters. If the retrained model is better, it replaces the original.

---

## Setup & Running

### Prerequisites

- Python 3.8+
- Node.js 16+

### Install Python dependencies

```bash
pip install scikit-learn pandas numpy joblib flask
```

### Step 1 — Train the model

```bash
python train_model.py
```

This will:
- Load and process the dataset
- Train and compare 3 models
- Select the best model
- Save model artifacts to `model/`

### Step 2 — Start Flask ML API

```bash
python app.py
```

Flask runs on `http://localhost:5000`

### Step 3 — Start Node.js Service

```bash
cd api
node server.js
```

Node.js runs on `http://localhost:3000`

---

## API Usage

### Predict Delivery Time

**Endpoint:** `POST http://localhost:3000/api/predict`

**Request body:**

```json
{
  "source_city": "Mumbai",
  "destination_city": "Pune",
  "parcel_weight_kg": 2.5,
  "distance_km": 148,
  "weather_condition": "Clear",
  "vehicle_type": "Bike"
}
```

**Response:**

```json
{
  "source_city": "Mumbai",
  "destination_city": "Pune",
  "parcel_weight_kg": 2.5,
  "distance_km": 148,
  "weather_condition": "Clear",
  "vehicle_type": "Bike",
  "predicted_delivery_hours": 5.3,
  "estimated_delivery_days": 0.2,
  "confidence": "high",
  "message": "Your parcel from Mumbai to Pune is expected to arrive in approximately 5.3 hours."
}
```

### Valid Input Values

**GET** `http://localhost:3000/api/info`

### Health Check

**GET** `http://localhost:3000/api/health`

---

## Valid Input Values

| Field | Accepted Values |
|---|---|
| `source_city` | Ahmedabad, Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata, Lucknow, Mumbai, Pune, Surat |
| `destination_city` | Same as above + Agra, Aurangabad, Bhopal, Bhubaneswar, Chandigarh, Coimbatore, Goa, Gurgaon, Guwahati, Kanpur, Ludhiana, Madurai, Mangalore, Mysore, Nagpur, Nashik, Patna, Pondicherry, Rajkot, Ranchi, Surat, Thane, Varanasi, Vijayawada, Warangal |
| `weather_condition` | `Clear`, `Cloudy`, `Rainy` |
| `vehicle_type` | `Bike`, `Van`, `Truck` |

---

## Deployment Notes

To deploy on a cloud platform (Render / Railway / Heroku):

1. Push this repo to GitHub
2. Set environment variable `ML_API_URL` in Node.js service to point to the deployed Flask URL
3. Flask app: set `PORT` env variable; already reads `os.environ.get("PORT", 5000)`
4. Node.js: set `PORT` and `ML_API_URL` env variables

Example:
```
ML_API_URL=https://your-flask-app.onrender.com
```

---

## Notes

- All libraries used are open-source (scikit-learn, Flask, pandas, numpy)
- Model artifacts are serialized with joblib for fast loading
- Node.js service uses zero external npm dependencies (built-in `http` module only)
