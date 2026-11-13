"""
Canonical driver series names.

Sources write these names into ``driver_data`` / ``forecast_snapshots`` and the
feature pipeline reads them. Units are in the name. ``*_forecast_*`` series are
values that were knowable before the target hour; ``*_actual_*`` series are
measurements and must only be used as lagged features.
"""

# Fingrid (Finland)
WIND_FORECAST_MW = "wind_forecast_mw"
WIND_ACTUAL_MW = "wind_actual_mw"
WIND_CAPACITY_MW = "wind_capacity_mw"
SOLAR_FORECAST_MW = "solar_forecast_mw"
CONSUMPTION_FORECAST_MW = "consumption_forecast_mw"
CONSUMPTION_ACTUAL_MW = "consumption_actual_mw"
PRODUCTION_FORECAST_MW = "production_forecast_mw"
NUCLEAR_ACTUAL_MW = "nuclear_actual_mw"
HYDRO_ACTUAL_MW = "hydro_actual_mw"
NET_IMPORT_ACTUAL_MW = "net_import_actual_mw"

# Open-Meteo weather (Finland aggregates)
TEMPERATURE_FORECAST_C = "temperature_forecast_c"  # population-weighted
WIND_SPEED_100M_FORECAST_MS = "wind_speed_100m_forecast_ms"  # wind-farm-area mean
SOLAR_RADIATION_FORECAST_WM2 = "solar_radiation_forecast_wm2"

# ENTSO-E
LOAD_FORECAST_MW = "load_forecast_mw"
WIND_FORECAST_ENTSOE_MW = "wind_forecast_entsoe_mw"
SOLAR_FORECAST_ENTSOE_MW = "solar_forecast_entsoe_mw"
HYDRO_RESERVOIR_NO_MWH = "hydro_reservoir_no_mwh"
HYDRO_RESERVOIR_SE_MWH = "hydro_reservoir_se_mwh"

# Bidding zones stored in spot_prices_raw
ZONES = ("FI", "SE1", "SE3", "EE")
