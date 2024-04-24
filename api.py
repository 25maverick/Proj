from fastapi import FastAPI
from typing import List
import pickle

# Load your model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Initialize FastAPI app
app = FastAPI()

# Load your model
model_path = '/users/weissyuan/M378/Model/Model/20240422T204234Z-5555a/Model.joblib'
model = load_model(model_path)

# Define prediction endpoint
@app.post("/predict/")
async def predict(data: List[List[float]]):
    predictions = model.predict(data)
    return {"predictions": predictions}

