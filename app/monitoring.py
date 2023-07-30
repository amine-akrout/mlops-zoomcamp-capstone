"""
Module for monitoring the API
"""

from datetime import datetime, timedelta

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from pymongo import MongoClient

client = MongoClient("mongodb://mongo:27017/")

db = client["credit_card_default"]
collection = db["predictions"]


# process the data
def prepare_data(data: pd.DataFrame):
    """This function is used to prepare the data for training"""
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
    return data.sample(1000).reset_index(drop=True)


def get_last_30_days_data():
    """Function to get the last 30 days data"""
    # get the current date
    today = datetime.now()
    # get the date 30 days ago
    thirty_days_ago = today - timedelta(days=30)
    # get the data from the last 30 days
    data = collection.find({"created_at": {"$gte": thirty_days_ago, "$lt": today}})
    # convert the data into a dataframe
    data = pd.DataFrame(list(data))
    return data


def get_reference_data():
    """Function to get the reference data"""
    data = pd.read_excel(
        "data/ref_data.xls",
        skiprows=1,
        index_col=0,
    )
    data = prepare_data(data)
    return data


def generate_dashboard():
    """Generate dashboard"""
    dasboard_name = "static/drift.html"
    data_drift_dashboard = Report(metrics=[DataDriftPreset()])

    reference_data = get_reference_data()
    current_data = get_last_30_days_data()

    current_data.rename(
        columns={"prediction": "target"},
        inplace=True,
    )

    column_mapping = ColumnMapping("target", "prediction")

    data_drift_dashboard.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    data_drift_dashboard.save_html(dasboard_name)
    print(f"Dashboard saved to {dasboard_name}")
    return dasboard_name
