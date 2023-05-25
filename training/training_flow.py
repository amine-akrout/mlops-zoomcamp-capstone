"""Training flow for the model."""
import os
import logging
from dotenv import load_dotenv
import pandas as pd
from minio import Minio
from minio.error import S3Error
from tempfile import TemporaryDirectory

import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from etl.get_data import download_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)

mlflow.tracking.set_tracking_uri("http://localhost:5000")
mlflow.tracking.set_registry_uri("sqlite:///mlflow.db")


mlflow.set_experiment("credit_card_default")


# read the env variables
load_dotenv()


def get_env_variable():
    minio_host = os.getenv("MINIO_HOST")
    minio_port = os.getenv("MINIO_PORT")
    minio_root_user = os.getenv("AWS_ACCESS_KEY_ID")
    minio_root_password = os.getenv("AWS_SECRET_ACCESS_KEY")
    data_bucket = os.getenv("DATA_BUCKET")
    return minio_host, minio_port, minio_root_user, minio_root_password, data_bucket


(
    minio_host,
    minio_port,
    minio_root_user,
    minio_root_password,
    data_bucket,
) = get_env_variable()


# create a minio client
def create_minio_client():
    """This function is used to create a minio client"""
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


minio_client = create_minio_client()

download_data(minio_client)


# read the data from minio
def read_data():
    """This function is used to read the data from minio"""
    # check if the parquet data already exists
    minio_client = create_minio_client()
    object_name = "credit_card.parquet"
    logging.info("Reading data from minio ...")
    with TemporaryDirectory() as tmpdirname:
        minio_client.fget_object(data_bucket, object_name, tmpdirname + "/data.parquet")
        data = pd.read_parquet(tmpdirname + "/data.parquet")
    return data


data = read_data()


# prepare the data for training
def prepare_data(data):
    # keep only SEX, AGE, PAY_0, PAY_2, PBILL_AMT1, PAY_AMT1, default payment next month
    data = data[
        [
            "SEX",
            "AGE",
            "PAY_0",
            "PAY_2",
            "BILL_AMT1",
            "PAY_AMT1",
            "default payment next month",
        ]
    ]
    # rename the columns
    data.rename(columns={"default payment next month": "target"}, inplace=True)
    # return random sample of 1000 rows
    return data.sample(1000)


data = prepare_data(data)


# split the data into train and test
def split_data(data):
    """This function is used to split the data into train and test"""
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data(data)


# train the model
def train_model(model, X_train, y_train, X_test, y_test):
    """This function is used to train the model"""
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        verbose=True,
    )
    with mlflow.start_run() as run:
        model.fit(
            X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, verbose=20
        )
        train_mertics = model.get_best_score()["learn"]
        test_mertics = model.get_best_score()["validation"]
        mlflow.log_metrics(train_mertics)
        mlflow.log_metrics(test_mertics)
        mlflow.catboost.log_model(model, "catboost-model")
        mlflow.end_run()


if __name__ == "__main__":
    load_dotenv()
    (
        minio_host,
        minio_port,
        minio_root_user,
        minio_root_password,
        data_bucket,
    ) = get_env_variable()
    download_data(minio_client)
