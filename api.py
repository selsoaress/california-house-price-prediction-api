from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow

app = FastAPI()
mlflow.set_tracking_uri("sqlite:///mlflow.db")

prod_model = mlflow.sklearn.load_model("models:/Modelo_Equilibrado/1")

class HouseInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float

@app.post("/predict")
def serve_prediction(input_data : HouseInput):
    data_dict = input_data.dict()
    df_input = pd.DataFrame([data_dict])
    prediction = prod_model.predict(df_input)
    return {'predicted_price': float(prediction[0])}