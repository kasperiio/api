"""
Driver data sources for forecasting.

Every source returns pandas DataFrames in one of two canonical shapes:

- Driver frame: columns ``timestamp`` (tz-aware UTC), ``series`` (str, see
  ``app.forecasting.series``), ``value`` (float).
- Price frame: columns ``timestamp`` (tz-aware UTC), ``zone`` (str, e.g. "FI"),
  ``price_eur_mwh`` (float, no VAT), ``resolution_minutes`` (int, 60 or 15).
"""
