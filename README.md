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

### Price forecast

A self-learning 7-day forecast of Finnish day-ahead prices (`/forecast`), with p10/p50/p90 bands.

- **Drivers:** wind, consumption, solar and nuclear from Fingrid; load/wind forecasts, neighbouring zone prices (SE1, SE3, EE) and Nordic hydro reservoirs from ENTSO-E; weather forecasts from Open-Meteo.
- **Additional history:** Open-Meteo Previous Runs supplies weather forecasts at their original lead times where available, replacing simulated forecast-error noise. ENTSO-E outage revisions are selected as of the forecast issue time. EEX's free EUA primary-auction reports provide a carbon-price proxy, available to the model from the following day and carried forward for at most a week. Historical data missing from these sources falls back to the existing inputs.
- **Outage capacity:** transmission notices preserve reported available MW. These values are the minimum among active notices for a border, not total border transfer capacity. Unavailable MW is only calculated when nominal capacity is supplied.
- **Outage history limitation:** the tested ENTSO-E backfill returned only the latest revision of each notice. Notices published after a historical issue time are excluded, which prevents leakage but leaves gaps in what can be reconstructed. Hourly ingestion retains revisions as they arrive going forward.
- **Transfer capacity:** day-ahead estimated net transfer capacity into Finland is requested separately for EE, SE1 and SE3 and masked beyond D+1. Some borders/periods have no published NTC. As with other archived day-ahead drivers, historical values may include revisions; captured issue-time snapshots are preferred.
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

To skip the API backfill on another machine, copy the history from one that already has it. Price and driver history, archived weather and outage revisions are copied; the target merges them without overwriting existing rows, then trains its own model on start:
```sh
# where the backfill ran
PYTHONPATH=. python scripts/history_transfer.py export --out history.db
# on the target, with history.db copied into its data/ volume, before starting the app
docker compose run --rm app python scripts/history_transfer.py import data/history.db
docker compose up -d
```
Import before the app's first start with forecasting enabled; otherwise its startup job is already backfilling from the APIs. Imported history skips most bulk price/driver downloads; additional weather and outage sources may still refresh historical windows for gaps and revisions.

Run the app with a single uvicorn worker, otherwise each worker runs the scheduled jobs.

Compare the median model against spike/regime mixtures, or compare source additions on identical historical evaluation inputs, without activating any model:
```sh
python scripts/forecast_experiments.py --db data/electricity_prices.db --out experiments/models --experiment models
python scripts/forecast_experiments.py --db data/electricity_prices.db --out experiments/sources --experiment sources
```
The source comparison evaluates every variant on real archived weather where present; its baseline is trained with simulated weather errors. Results are separate from the production replay's calibrated quantile metrics. Predictions and MAE summaries are written to the selected output directory. Gas prices are not yet integrated; Trading Hub Europe's public VTP index is a potential gas proxy, subject to a working official endpoint.

`/forecast/drivers` returns p50 in EUR/MWh; SHAP contributions and the base value are in `asinh(price / 10)` space, identified by `contribution_unit`. Spike models expose `explanation_method=base_model_shap_plus_spike_adjustment`: feature contributions explain the base model and a separate `spike_adjustment` reconciles them to the final median. That adjustment is not a feature SHAP value.

Experiments on 2023-01-01–2026-11-13 history (median only; these are not calibrated production-forecast metrics):

| Archived-weather model experiment | Overall MAE, EUR/MWh | Last-year MAE | Spike-hour MAE (>100 EUR/MWh) |
|---|---:|---:|---:|
| Single model | 26.79 | 26.42 | 77.62 |
| Spike mixture | 26.53 | 26.31 | 72.81 |
| Baltic-coupling mixture | 26.83 | 26.72 | 76.67 |

This initial median-only run used 191,611 evaluation rows, a 180-day warmup and 28-day retraining blocks. The outage backfill was still partial when its input frame was built.

A separate source comparison after the outage backfill, on the same dates and retraining cadence, found no robust improvement from adding every source: the baseline trained on simulated weather errors scored 27.13 overall / 26.12 last-year MAE, versus 27.06 / 27.46 with archived weather, carbon and outages. That run used the completed outage history and a stricter radiation-run availability cutoff, so its absolute MAEs should not be compared directly with the earlier model experiment. No experimental model was activated. The existing served model configuration also beat larger models, Huber/L2 losses and recency decay; daily retraining now uses uniform weights, consistent with bootstrap training.

The full spike model retains the base quantile bounds, expands them to contain its new median when necessary, and uses the existing rolling band calibration. A later replay on the completed history, with calibration fitted only from outcomes before each issue, scored 177,571 rows after a 90-day calibration warmup:

| Calibrated replay | Overall MAE, EUR/MWh | Last-year MAE | Spike-hour MAE | p10–p90 coverage | Mean interval width, EUR/MWh |
|---|---:|---:|---:|---:|---:|
| Base quantile model | 26.34 | 27.30 | 78.75 | 79.2% | 82.45 |
| Spike model | 25.93 | 26.71 | 72.47 | 79.5% | 82.39 |

The spike model passed the predeclared median, recent-year, spike-hour and band-quality checks. Its weighted conditional medians are an empirical predictor, not the mathematical median of a mixture distribution. Weekly and drift retraining compare it with the base challenger on the same recent holdout, then compare the selected challenger with the active model. Promotion also checks finite ordered bands, coverage, width and interval score; an MAE improvement alone is insufficient. A spike fit with too few examples falls back to the base model.

The latest scratch-database comparison (2026-10-16–2026-11-13 holdout) chose the base challenger: 29.07 EUR/MWh versus 29.45 for the spike variant. Neither replaced the active model. The incumbent had already trained on that holdout, so its 22.53 MAE is not an independent out-of-sample benchmark; promotion remains conservative until fresh outcomes accumulate.

Reproduce the band validation or request a manual challenger comparison (the second command can activate a winning model):
```sh
python scripts/validate_spike.py --db data/electricity_prices.db --out experiments/spike-bands
python scripts/retrain_forecast.py --db data/electricity_prices.db --spike
```

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
