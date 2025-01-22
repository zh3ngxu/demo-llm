"""
fastapi script to predict MPG based on car attributes

To run the API:
pip3 install fastapi uvicorn numpy pydantic scikit-learn
python3 -m uvicorn mpg_api:app --reload

export model py code
import pickle
with open('model.pkl', 'wb') as files:
    pickle.dump(model, files)

send post request to /predict with body param
{
    "cylinders": 307,
    "displacement": 130.0,
    "horsepower": 3500.0,
    "weight": 3504,
    "acceleration": 12,
    "model_year": 70
}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Load the trained model
MODEL_PATH = "./model.pkl"  # Path to your trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

scaler = StandardScaler()

# FastAPI app instance
app = FastAPI(title="MPG Prediction API", description="API to predict MPG based on car attributes", version="1.0")

# Request body model
class CarAttributes(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: int
    acceleration: float
    model_year: int

# Endpoint to predict mpg
@app.post("/predict")
async def predict_mpg(car: CarAttributes):
    try:
        input_data = np.array([[car.cylinders, car.displacement, car.horsepower, car.weight, car.acceleration, car.model_year,1]])
        
        input_data_scaled = scaler.fit_transform(input_data)

        prediction = model.predict(input_data_scaled)

        return {"mpg_prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the MPG Prediction API!"}
