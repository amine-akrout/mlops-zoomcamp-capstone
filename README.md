# Credit Card Default Prediction

This project aims to predict whether a credit card holder will default on their payment next month based on their demographic and payment history data.

## Problem statement

Credit card companies need to be able to predict which customers are likely to default on their payments in order to minimize their financial losses. This project aims to build a machine learning model that can accurately predict whether a credit card holder will default on their payment next month based on their demographic and payment history data.

## Dataset

The dataset used in this project is the [UCI Credit Card Default Payment dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). It contains information on credit card holders in Taiwan from April 2005 to September 2005, including demographic data, payment history, and default payment status.

## Approach

The project follows the following approach:

1. Model training and evaluation : A machine learning model is trained evaluated
2. Model tracking: The trained model is tracked using MLflow.
3. Workflow automation: The model training process is automated using Prefect.
4. Model deployment: The trained model is deployed in a production environment for real-world use.
5. Model monitoring: The deployed model is monitored to ensure that it continues to perform well.


## Requirements

The project requires the following dependencies:

- Python 3.6 or higher
- Docker

## Usage

To run the project, follow these steps:

1. Clone the repository: `git clone https://github.com/amine-akrout/mlops-zoomcamp-capstone.git`
2. Setup the environment: `Make setup`
3. Run docker-compose: `docker-compose up`
This will start the MLflow server, the Prefect server, the Prefect agent and the fastapi (prdiction service) server.
