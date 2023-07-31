FROM prefecthq/prefect:2.11.0-python3.9

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# # Copy local code to the container image.
# WORKDIR /app

COPY requirements.txt .

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
