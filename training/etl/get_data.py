"""
This file is used to download the data from kaggle and updload as parquet file to Minio
"""

import logging
import os
import shutil
from tempfile import TemporaryDirectory

import pandas as pd
from dotenv import load_dotenv


from minio.error import S3Error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s]: %(message)s"
)


load_dotenv()


def check_object_exists(minio_client, bucket_name, object_name):
    """This function is used to check if the object exists in the bucket"""
    try:
        minio_client.stat_object(bucket_name, object_name)
        return True
    except S3Error:
        return False


# download the data from kaggle and upload to minio as parquet file
def download_data(minio_client):
    """
    This function is used to download the data from kaggle and upload to minio as parquet file
    """
    # check if the parquer data already exists
    bucket_name = os.getenv("DATA_BUCKET")
    object_name = "credit_card.parquet"
    if not minio_client.bucket_exists(bucket_name):
        logging.info("Creating bucket %s", bucket_name)
        minio_client.make_bucket(bucket_name)
        logging.info("Bucket %s created successfully", bucket_name)
    exists = check_object_exists(minio_client, bucket_name, object_name)
    if exists:
        logging.info("Data already exists in minio ...")
        return
    logging.info("Downloading data from UCI ...")
    with TemporaryDirectory() as tmpdirname:
        data_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
            "default%20of%20credit%20card%20clients.xls"
        )
        data = pd.read_excel(data_url, header=1)
        logging.info("Data downloaded successfully ...")
        logging.info("Converting data to parquet")
        data.to_parquet(tmpdirname + "/credit_card.parquet")
        try:
            logging.info("Uploading data to minio ...")
            minio_client.fput_object(
                bucket_name=bucket_name,
                object_name="credit_card.parquet",
                file_path=tmpdirname + "/credit_card.parquet",
            )
            logging.info("Data uploaded successfully ...")
        except S3Error as err:
            logging.error(err)
        shutil.rmtree(tmpdirname)
