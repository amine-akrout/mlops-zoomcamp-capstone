FROM prefecthq/prefect:2.11.0-python3.9

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# # Copy local code to the container image.
# WORKDIR /app

COPY requirements.txt .

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Run the web service on container startup using gunicorn

# CMD ["python", "deploy.py"]
# CMD sleep 60 && python training_flow.py

# CMD ["python", "training_flow.py"]

CMD sleep 60 && python deploy.py
