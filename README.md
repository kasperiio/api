# Kasperio API

This application provides API for different usecases (Electricity).

## Features

### Electricity

- Retrieve electricity prices for a specified time range and store it into a SQLite DB.
- Fetch missing data from an external source (ENTSO-E API).
- Find the cheapest consecutive or non-consecutive hours.
- Find the current price.
- Calculate the hourly price ratio to daily average

### Price forecast

A self-learning 7-day forecast of Finnish day-ahead prices (`/forecast`), with p10/p50/p90 bands.

- **Drivers:** wind, consumption, solar and nuclear from Fingrid; load/wind forecasts, neighbouring zone prices (SE1, SE3, EE) and Nordic hydro reservoirs from ENTSO-E; weather forecasts from Open-Meteo.
- **Model:** LightGBM quantile regression. On first start the app backfills data from 2023 and replays history week by week to train the first model and measure its accuracy.
- **Self-learning:** the app runs scheduled jobs in-process (Europe/Helsinki time):

  | When | Job |
  |---|---|
  | Hourly | Save a forecast snapshot |
  | Daily 14:30 | Score past predictions against actual prices, update bias correction and band calibration, check for drift |
  | Daily 14:45 | Publish a new forecast |
  | Weekly, or on drift | Retrain; the new model is only activated if it beats the current one |
- **Endpoints:** `/forecast`, `/forecast/accuracy`, `/forecast/drivers`, `/forecast/status`.

Environment variables:

| Variable | Purpose |
|---|---|
| `ENTSOE_API_KEY` | ENTSO-E Transparency API key |
| `FINGRID_API_KEY` | Fingrid Open Data API key (data.fingrid.fi) |
| `FORECAST_ENABLED` | Set to `false` to disable the forecast scheduler (default `true`) |

On first start the app backfills history from 2023, which takes a while because of API rate limits; it resumes if interrupted. It can also be run up front:
```sh
PYTHONPATH=. python scripts/backfill.py --start 2023-01-01
```

To skip the API backfill on another machine, copy the history from one that already has it. Only the price and driver history is copied; the target merges it in without overwriting existing rows, then trains its own model on start:
```sh
# where the backfill ran
PYTHONPATH=. python scripts/history_transfer.py export --out history.db
# on the target, with history.db copied into its data/ volume, before starting the app
docker compose run --rm app python scripts/history_transfer.py import data/history.db
docker compose up -d
```
Import before the app's first start with forecasting enabled; otherwise its startup job is already backfilling from the APIs. The app then only fetches the data newer than the export.

Run the app with a single uvicorn worker, otherwise each worker runs the scheduled jobs.

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
    uvicorn main:app --reload --log-config log_config.yaml
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