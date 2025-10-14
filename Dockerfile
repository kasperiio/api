# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create a virtual environment (we're already in a container)
RUN poetry config virtualenvs.create false

# Install dependencies only (don't install the project itself yet)
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy the rest of the application code into the container
COPY . .

# Install the project itself now that all files are present
RUN poetry install --no-interaction --no-ansi --only-root

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "log_config.yaml"]