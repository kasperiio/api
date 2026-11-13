# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==2.2.1

# Copy poetry configuration files
COPY pyproject.toml poetry.lock ./

# Configure Poetry to not create a virtual environment (we're already in a container)
RUN poetry config virtualenvs.create false

# Install dependencies only (we don't need to install the project as a package)
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application (single worker: the forecasting scheduler runs in-process)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-config", "log_config.yaml"]
