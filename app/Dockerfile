# Use the official Python image as the base image
FROM python:3.9-slim


# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the container at /app
COPY . /app

# Expose the port on which the FastAPI app will run (make sure it matches the FastAPI app port)
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
