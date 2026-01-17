


import pandas as pd
import numpy as np
import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# LOAD & CLEAN DATASET
# =========================
data = pd.read_csv("house_data.csv")

# Menghapus data duplikat
data = data.drop_duplicates()

# Menghapus missing value
data = data.dropna()

# =========================
# FEATURE & TARGET
# =========================
X = data.drop("price", axis=1)
y = data["price"]

# =========================
# PREPROCESSING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
print("R2 Score :", r2_score(y_test, y_pred))
print("RMSE     :", mean_squared_error(y_test, y_pred, squared=False))

# =========================
# SAVE MODEL
# =========================
with open("model_rumah.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("Model berhasil disimpan")

# =========================
# FASTAPI
# =========================
app = FastAPI(title="API Prediksi Harga Rumah")

class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    floors: float

@app.get("/")
def home():
    return {"message": "API Prediksi Harga Rumah Aktif"}

@app.post("/predict")
def predict_price(data: HouseInput):
    with open("model_rumah.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    input_data = np.array([[ 
        data.bedrooms,
        data.bathrooms,
        data.sqft_living,
        data.floors
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return {
        "predicted_price": float(prediction[0])
    }




