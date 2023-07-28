"""
Module to run the app
"""
# pylint: disable=E0401,E0611,R0903
import os
import logging
import warnings
from fastapi import FastAPI
from loguru import logger
import pandas as pd
from pydantic import BaseModel
from minio import Minio
from minio.error import S3Error
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


def get_env_variable():
    """This function is used to get the environment variables"""
    minio_host = os.getenv("MINIO_HOST")
    minio_port = os.getenv("MINIO_PORT")
    minio_root_user = os.getenv("AWS_ACCESS_KEY_ID")
    minio_root_password = os.getenv("AWS_SECRET_ACCESS_KEY")
    model_bucket = os.getenv("MLFLOW_BUCKET")
    return minio_host, minio_port, minio_root_user, minio_root_password, model_bucket


_ = load_dotenv()

(
    minio_host,
    minio_port,
    minio_root_user,
    minio_root_password,
    model_bucket,
) = get_env_variable()


# create a minio client
def create_minio_client(
    minio_host: str, minio_port: str, minio_root_user: str, minio_root_password: str
) -> Minio:
    """This function is used to create a minio client
    Args:
        minio_host (str): minio host
        minio_port (str): minio port
        minio_root_user (str): minio root user
        minio_root_password (str): minio root password
    Returns:
        Minio: minio client
    """
    minio_client = Minio(
        endpoint=minio_host + ":" + minio_port,
        access_key=minio_root_user,
        secret_key=minio_root_password,
        secure=False,
    )
    try:
        minio_client.list_buckets()
    except S3Error as exc:
        logging.error("error occurred %s", exc)
        raise
    return minio_client


@app.on_event("startup")
def start_load_model():
    """Function to load the model on startup"""
    global model
    logger.info("Loading the model...")
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_tracking_uri("http://mlflow:5000")

    # # experiment = mlflow.get_experiment_by_name("default_prediction")
    # # run_id = experiment.latest_run_id
    # run_id = "f6346a8beafe4db5a329d727c9512b67"
    # logged_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/catboost-model")
    # model = mlflow.pyfunc.load_model(logged_model)

    logged_model = "runs:/f6346a8beafe4db5a329d727c9512b67/catboost-model"
    # Load model as a PyFuncModel.
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
