"""
Module to run the app
"""
# pylint: disable=E0401,E0611,R0903
import logging
import warnings
from fastapi import FastAPI
from loguru import logger
import pandas as pd
from pydantic import BaseModel
import mlflow
from dotenv import load_dotenv


warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)


app = FastAPI(routes=[], title="Credit Card Default Prediction API", version="0.0.1")


class User(BaseModel):
    """User class"""

    sex: int = None
    age: int = None
    pay_0: int = None
    pay_2: int = None
    bill_amt1: int = None
    pay_amt1: int = None


_ = load_dotenv()


@app.on_event("startup")
def start_load_model():
    """Function to load the model on startup"""
    global model
    logger.info("Loading the model...")
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_tracking_uri("http://mlflow:5000")

    runs = mlflow.search_runs(experiment_ids=["1"])
    best_run = runs.sort_values("metrics.AUC", ascending=False).iloc[0]
    best_run_id = best_run["run_id"]
    logged_model = f"runs:/{best_run_id}/catboost-model"
    model = mlflow.pyfunc.load_model(logged_model)


@app.get("/")
def read_root():
    """Function to get the root"""
    return {"message": "Welcome to the Credit Card Default Prediction API"}


@app.post("/predict")
def predict(user: User):
    """Function to predict the default payment"""
    logger.info("Predicting the default payment")
    user_dict = user.dict()
    user_df = pd.DataFrame([user_dict])
    prediction = model.predict(user_df)
    if prediction[0] == 0:
        prediction = "Not Default"
    else:
        prediction = "Default"
    return {"prediction": prediction}


@app.get("/health")
def health():
    """Function to check the health of the app"""
    return {"status": "ok"}
