"""Training flow for the model."""
import logging
import os
from tempfile import TemporaryDirectory

import mlflow
import mlflow.xgboost
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from sklearn.model_selection import train_test_split

from etl.get_data import download_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)

# mlflow.tracking.set_tracking_uri("http://localhost:5000")
# mlflow.tracking.set_registry_uri("sqlite:///mlflow.db")


def get_env_variable():
    """This function is used to get the environment variables"""
    minio_host = os.getenv("MINIO_HOST")
    minio_port = os.getenv("MINIO_PORT")
    minio_root_user = os.getenv("AWS_ACCESS_KEY_ID")
    minio_root_password = os.getenv("AWS_SECRET_ACCESS_KEY")
    data_bucket = os.getenv("DATA_BUCKET")
    return minio_host, minio_port, minio_root_user, minio_root_password, data_bucket


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


# read the data from minio
def read_data(
    minio_client: Minio, data_bucket: str, object_name: str = "credit_card.parquet"
):
    """This function is used to read the data from minio"""
    # check if the parquet data already exists
    logging.info("Reading data from minio ...")
    with TemporaryDirectory() as tmpdirname:
        minio_client.fget_object(data_bucket, object_name, tmpdirname + "/data.parquet")
        data = pd.read_parquet(tmpdirname + "/data.parquet")
    return data


# prepare the data for training
def prepare_data(data: pd.DataFrame):
    """This function is used to prepare the data for training
    Args:
        data (pd.DataFrame): data
    Returns:
        pd.DataFrame: data
    """
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
    data.rename(
        columns={
            "default payment next month": "target",
            "SEX": "sex",
            "AGE": "age",
            "PAY_0": "pay_0",
            "PAY_2": "pay_2",
            "BILL_AMT1": "bill_amt1",
            "PAY_AMT1": "pay_amt1",
        },
        inplace=True,
    )
    # return random sample of 1000 rows
    return data.sample(1000)


# split the data into train and test
def split_data(data: pd.DataFrame):
    """This function is used to split the data into train and test
    Args:
        data (pd.DataFrame): data
    Returns:
        pd.DataFrame: train features
        pd.DataFrame: test features
        pd.DataFrame: train target
        pd.DataFrame: test target
    """
    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


# train the model
def train_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
):
    """This function is used to train the model
    Args:
        x_train (pd.DataFrame): train features
        y_train (pd.DataFrame): train target
        x_test (pd.DataFrame): test features
        y_test (pd.DataFrame): test target
    """
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        eval_metric=["AUC", "Accuracy", "Recall", "Precision", "F1"],
        verbose=True,
    )
    with mlflow.start_run() as run:
        model.fit(
            x_train, y_train, eval_set=(x_test, y_test), use_best_model=True, verbose=20
        )
        train_mertics = model.get_best_score()["learn"]
        test_mertics = model.get_best_score()["validation"]
        mlflow.log_metrics(train_mertics)
        mlflow.log_metrics(test_mertics)
        mlflow.catboost.log_model(model, "catboost-model")
        mlflow.end_run()


def training_flow():
    """Main flow of the training"""
    _ = load_dotenv()
    (
        minio_host,
        minio_port,
        minio_root_user,
        minio_root_password,
        data_bucket,
    ) = get_env_variable()
    minio_client = create_minio_client(
        minio_host, minio_port, minio_root_user, minio_root_password
    )
    download_data(minio_client)
    data = read_data(minio_client, data_bucket)
    prepared_data = prepare_data(data)
    x_train, x_test, y_train, y_test = split_data(prepared_data)
    mlflow.set_experiment("credit_card_default")
    train_model(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    training_flow()
