# app.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("D:\\ML BOOTCAMP\\titanic_ml\\models\\random_forest_best_model.pkl")

@app.get("/")
def home():
    return {"message": "Titanic Prediction API is running"}

@app.post("/predict")
def predict(Pclass: int, Sex: str, Age: float, SibSp: int, Parch: int, Fare: float, Embarked: str):
    df = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked
    }])
    pred = model.predict(df)[0]
    return {"Survived": int(pred)}
