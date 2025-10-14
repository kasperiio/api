# Kasperio API

This application provides API for different usecases (Electricity).

## Features

### Electricity

- Retrieve electricity prices for a specified time range and store it into a SQLite DB.
- **Data provider:**
  - ENTSO-E API (requires free API key)
- **Built-in rate limiting** using `httpx` to respect API limits and prevent bans
  - ENTSO-E: 10 requests per second
- **Smart data handling:**
  - Automatic chunking of large date ranges (30-day chunks)
  - NULL placeholders for genuinely unavailable data to prevent repeated fetching
  - Deduplication using sets for efficient processing
- Find the cheapest consecutive or non-consecutive hours.
- Find the current price.
- Calculate the hourly price ratio to daily average

## Installation

### Prerequisites

- Python 3.10+
- Poetry (recommended) or pip
- Docker (optional, for Docker usage)

### Local Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kasperiio/api.git
    cd api
    ```

2. Install dependencies using Poetry (recommended):
    ```sh
    poetry install
    poetry shell
    ```

    Or using pip:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Configure environment variables:
    ```sh
    cp .env.example .env
    # Edit .env and add your ENTSO-E API key (required)
    ```

    To get an ENTSO-E API key:
    - Register at https://transparency.entsoe.eu/
    - Request an API key (it's free)
    - Add it to your `.env` file as `ENTSOE_API_KEY=your_key_here`

## Usage

### Local Usage

1. Start the application:

    With Poetry:
    ```sh
    poetry run uvicorn main:app --reload --log-config log_config.yaml
    ```

    Or without Poetry:
    ```sh
    uvicorn main:app --reload --log-config log_config.yaml
    ```

2. Access the API documentation at `http://127.0.0.1:8000/docs`.

### Data Provider

The API uses ENTSO-E (European Network of Transmission System Operators for Electricity) as the data source:

- **ENTSO-E API**:
  - Requires free API key (set in `.env`)
  - Rate limited to 10 requests per second
  - Provides electricity prices with 25.5% VAT included
  - Automatic chunking for large date ranges (30-day chunks)
  - Smart handling of missing data:
    - Returns NULL for timestamps where data is genuinely unavailable
    - Prevents repeated fetching of unavailable data
    - Caches all responses in SQLite database for fast subsequent requests

**Note**: Make sure you have configured your ENTSO-E API key in the `.env` file before starting the application.

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