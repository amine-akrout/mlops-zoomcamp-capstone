"""
Module to test the training flow module
"""

# pylint: disable=W0621, E0401, C0413
import os
import sys

# Add the parent directory of the training directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import pytest

from training.training_flow import prepare_data, split_data


@pytest.fixture
def test_data():
    """
    Fixture to return test data

    Returns:
        pd.DataFrame: test data
    """
    return pd.DataFrame(
        {
            "SEX": [1, 2, 1, 2],
            "AGE": [30, 25, 35, 40],
            "PAY_0": [1, -1, 2, 0],
            "PAY_2": [0, -1, 1, 0],
            "BILL_AMT1": [1000, 2000, 3000, 4000],
            "PAY_AMT1": [500, 800, 1000, 1200],
            "default payment next month": [0, 1, 0, 1],
        }
    )


def test_prepare_data(test_data):
    """
    Test the prepare_data function

    Args:
        test_data (pd.DataFrame): test data
    """
    expected_columns = [
        "sex",
        "age",
        "pay_0",
        "pay_2",
        "bill_amt1",
        "pay_amt1",
        "target",
    ]
    expected_sample_size = min(1000, len(test_data))

    # Act
    prepared_data = prepare_data.fn(test_data)

    # Assert
    assert list(prepared_data.columns) == expected_columns
    assert len(prepared_data) == expected_sample_size


def test_split_data(test_data):
    """
    Test the split_data function

    Args:
        test_data (pd.DataFrame): test data
    """

    # Act
    prepared_data = prepare_data.fn(test_data)
    x_train, x_test, y_train, y_test = split_data.fn(prepared_data)

    # Assert
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_test) < len(test_data)
