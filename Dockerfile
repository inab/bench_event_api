# Use an official Python runtime as a parent image
FROM python:3.5-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY [ "requirements.txt","constraints.txt", "/app/" ]

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get -y install build-essential libpoppler-cpp-dev pkg-config python-dev && pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt