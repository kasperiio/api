# Kasperio API

This application provides API for different usecases (Electricity).

## Features

### Electricity

- Retrieve electricity prices for a specified time range and store it into a SQLite DB.
- Fetch missing data from an external source (ENTSO-E API).
- Find the cheapest consecutive or non-consecutive hours.
- Find the current price.
- Calculate the hourly price ratio to daily average

## Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for Docker usage)

### Local Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kasperiio/api.git
    cd api
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Local Usage

1. Start the application:
    ```sh
    uvicorn app.main:app --reload
    ```

2. Access the API documentation at `http://127.0.0.1:8000/docs`.

### Docker Usage

1. Modify the docker-compose.yaml to you likeing

21. Build and start the Docker containers:
    ```sh
    docker-compose up --build -d
    ```

3. Access the API documentation at `http://127.0.0.1:8000/docs`.

4. To stop the container:
    ```sh
    docker-compose down
    ```

## Contribution

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.