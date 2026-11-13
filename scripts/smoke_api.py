"""Check a fresh API instance with forecasting disabled and no external API keys."""

import argparse
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    deadline = time.monotonic() + 60
    while True:
        try:
            response = requests.get(base + "/docs", timeout=3)
            response.raise_for_status()
            break
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    for path, expected in (("/docs", 200), ("/forecast/status", 200),
                           ("/forecast/accuracy", 200), ("/forecast", 404)):
        response = requests.get(base + path, timeout=10)
        if response.status_code != expected:
            raise RuntimeError(f"{path}: expected {expected}, got {response.status_code}: {response.text}")
        if path == "/forecast/status":
            assert response.json()["scheduler"]["enabled"] is False
        print(f"{path}: {response.status_code}", flush=True)


if __name__ == "__main__":
    main()
