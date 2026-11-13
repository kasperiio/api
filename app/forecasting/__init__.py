"""
Electricity price forecasting package.

Collects price drivers (wind, consumption, weather, neighbouring prices...),
trains a self-correcting LightGBM quantile model and serves D+1..D+7 forecasts.
All timestamps are timezone-aware UTC; prices are EUR/MWh without VAT.
"""
