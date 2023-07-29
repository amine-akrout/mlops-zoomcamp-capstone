"""
Module to run the app
"""
# pylint: disable=E0401,E0611,R0903
import logging
import warnings
from datetime import datetime

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi_utils.tasks import repeat_every
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel
from pymongo import MongoClient

from monitoring import generate_dashboard

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)


client = MongoClient("mongodb://mongo:27017/")
db = client["credit_card_default"]
collection = db["predictions"]

app = FastAPI(routes=[], title="Credit Card Default Prediction API", version="0.0.1")


def save_prediction_to_db(user: dict, prediction: int):
    """Function to save the prediction to the database"""
    logger.info("Saving the prediction to the database")
    user["prediction"] = int(prediction)
    user["created_at"] = datetime.utcnow()
    collection.insert_one(user)


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
@repeat_every(seconds=10)  # run every 24 hours
async def start_load_model():
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
async def read_root():
    """Function to get the root"""
    return {"message": "Welcome to the Credit Card Default Prediction API"}


@app.post("/predict", tags=["prediction"])
async def predict(user: User, background_tasks: BackgroundTasks):
    """Function to predict the default payment"""
    logger.info("Predicting the default payment")
    user_dict = user.dict()
    user_df = pd.DataFrame([user_dict])
    prediction = model.predict(user_df)
    if prediction[0] == 0:
        pred_message = "Not Default"
    else:
        pred_message = "Default"
    # add background task to save the prediction to the database
    background_tasks.add_task(save_prediction_to_db, user_dict, prediction[0])

    return {"prediction": pred_message}


@app.get("/monitoring", tags=["dashboard"])
async def monitoring():
    """Function to generate the dashboard"""
    logger.info("Generating the dashboard")
    dashboard_location = generate_dashboard()
    return FileResponse(dashboard_location)


@app.get("/health", tags=["health"])
async def health():
    """Function to check the health of the app"""
    return {"status": "ok"}
