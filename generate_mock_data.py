"""
generate_mock_data.py
Generates 6 months of realistic synthetic hourly bike-sharing data
for 10 Valenbisi stations across Valencia, including weather features.
Outputs: data/stations.csv and data/hourly_demand.csv
"""

import numpy as np
import pandas as pd
import os
import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATIONS = 10
START_DATE = "2024-01-01"
END_DATE = "2024-06-30"
SEED = 42

# Station definitions: realistic Valencia coordinates
STATIONS = [
    {"station_id": 1,  "name": "Plaza del Ayuntamiento",      "lat": 39.4699, "lon": -0.3763, "capacity": 30, "type": "transport_hub"},
    {"station_id": 2,  "name": "Estación del Norte",          "lat": 39.4670, "lon": -0.3816, "capacity": 40, "type": "transport_hub"},
    {"station_id": 3,  "name": "UV Tarongers",                "lat": 39.4800, "lon": -0.3560, "capacity": 35, "type": "university"},
    {"station_id": 4,  "name": "UPV Campus",                  "lat": 39.4815, "lon": -0.3450, "capacity": 40, "type": "university"},
    {"station_id": 5,  "name": "Ciudad de las Artes",         "lat": 39.4543, "lon": -0.3535, "capacity": 35, "type": "tourist"},
    {"station_id": 6,  "name": "Playa Malvarrosa",            "lat": 39.4750, "lon": -0.3300, "capacity": 30, "type": "tourist"},
    {"station_id": 7,  "name": "Ruzafa",                      "lat": 39.4620, "lon": -0.3720, "capacity": 25, "type": "commercial"},
    {"station_id": 8,  "name": "Benimaclet",                  "lat": 39.4860, "lon": -0.3630, "capacity": 25, "type": "residential"},
    {"station_id": 9,  "name": "Av. Francia",                 "lat": 39.4625, "lon": -0.3450, "capacity": 30, "type": "commercial"},
    {"station_id": 10, "name": "Hospital La Fe",              "lat": 39.4438, "lon": -0.3830, "capacity": 30, "type": "public_service"},
]

# Base hourly demand profiles per station type (multipliers, 24 values for hours 0..23)
BASE_PROFILES = {
    "residential":   [0.03, 0.02, 0.02, 0.02, 0.03, 0.08, 0.25, 0.70, 0.90, 0.45,
                      0.35, 0.30, 0.30, 0.35, 0.40, 0.45, 0.55, 0.75, 0.85, 0.65,
                      0.40, 0.25, 0.15, 0.08],
    "university":    [0.02, 0.01, 0.01, 0.02, 0.04, 0.10, 0.25, 0.65, 0.95, 0.85,
                      0.65, 0.60, 0.55, 0.50, 0.55, 0.65, 0.60, 0.50, 0.45, 0.35,
                      0.20, 0.12, 0.08, 0.04],
    "commercial":    [0.02, 0.01, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.50, 0.75,
                      0.85, 0.80, 0.70, 0.65, 0.60, 0.65, 0.70, 0.80, 0.85, 0.75,
                      0.55, 0.30, 0.15, 0.06],
    "tourist":       [0.04, 0.03, 0.02, 0.02, 0.04, 0.08, 0.15, 0.25, 0.40, 0.60,
                      0.70, 0.70, 0.65, 0.55, 0.50, 0.55, 0.65, 0.75, 0.80, 0.70,
                      0.50, 0.30, 0.15, 0.08],
    "transport_hub": [0.06, 0.04, 0.03, 0.03, 0.06, 0.20, 0.45, 0.75, 0.90, 0.60,
                      0.50, 0.45, 0.45, 0.50, 0.55, 0.60, 0.65, 0.80, 0.85, 0.65,
                      0.40, 0.25, 0.15, 0.10],
    "public_service":[0.03, 0.02, 0.02, 0.02, 0.04, 0.10, 0.25, 0.60, 0.80, 0.70,
                      0.65, 0.60, 0.55, 0.50, 0.55, 0.60, 0.55, 0.45, 0.35, 0.25,
                      0.18, 0.12, 0.08, 0.04],
}

# Weekend demand profile modifiers: flatter, less commute-peaked
WEEKEND_PROFILES = {
    "residential":   [0.04, 0.03, 0.02, 0.02, 0.03, 0.06, 0.12, 0.20, 0.30, 0.40,
                      0.50, 0.55, 0.50, 0.45, 0.40, 0.38, 0.40, 0.50, 0.55, 0.50,
                      0.45, 0.35, 0.20, 0.10],
    "university":    [0.02, 0.01, 0.01, 0.01, 0.02, 0.04, 0.08, 0.12, 0.15, 0.18,
                      0.20, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08,
                      0.08, 0.06, 0.04, 0.03],
    "commercial":    [0.05, 0.03, 0.02, 0.02, 0.04, 0.08, 0.15, 0.25, 0.40, 0.60,
                      0.80, 0.90, 0.85, 0.75, 0.65, 0.60, 0.65, 0.75, 0.80, 0.85,
                      0.90, 0.70, 0.40, 0.15],
    "tourist":       [0.06, 0.04, 0.03, 0.03, 0.05, 0.10, 0.18, 0.30, 0.45, 0.65,
                      0.80, 0.85, 0.80, 0.70, 0.60, 0.55, 0.60, 0.70, 0.80, 0.75,
                      0.65, 0.45, 0.25, 0.12],
    "transport_hub": [0.06, 0.04, 0.03, 0.03, 0.06, 0.12, 0.20, 0.30, 0.40, 0.50,
                      0.55, 0.55, 0.50, 0.48, 0.45, 0.48, 0.50, 0.55, 0.55, 0.50,
                      0.45, 0.35, 0.22, 0.12],
    "public_service":[0.04, 0.03, 0.02, 0.02, 0.04, 0.08, 0.15, 0.20, 0.25, 0.35,
                      0.45, 0.50, 0.48, 0.40, 0.35, 0.30, 0.28, 0.28, 0.25, 0.22,
                      0.20, 0.15, 0.10, 0.06],
}

# Monthly seasonal demand multipliers (base = 1.0) — higher in spring/summer
MONTHLY_DEMAND = {1: 0.65, 2: 0.72, 3: 0.85, 4: 0.95, 5: 1.10, 6: 1.20}

# Base scale per station type (avg bikes checked out per hour at peak season)
BASE_SCALE = {
    "residential": 14,
    "university": 16,
    "commercial": 13,
    "tourist": 12,
    "transport_hub": 18,
    "public_service": 10,
}

# Valencia-specific holiday/event dates (2024)
SPECIAL_DATES = {
    "2024-01-01": ("new_year", 0.15),       # Año Nuevo
    "2024-01-06": ("reyes", 0.20),          # Día de Reyes
    "2024-03-15": ("fallas_start", 0.30),   # Plantà
    "2024-03-16": ("fallas", 0.25),
    "2024-03-17": ("fallas", 0.25),
    "2024-03-18": ("fallas", 0.20),
    "2024-03-19": ("fallas_end", 0.15),     # Cremà, San José
    "2024-03-28": ("semana_santa", 0.45),   # Jueves Santo
    "2024-03-29": ("semana_santa", 0.40),   # Viernes Santo
    "2024-04-01": ("easter_monday", 0.50),  # Lunes de Pascua
    "2024-05-01": ("labor_day", 0.30),      # Día del Trabajador
}


def generate_stations_csv(output_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(STATIONS)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "stations.csv")
    df.to_csv(path, index=False)
    print(f"[OK] {path} — {len(df)} stations")
    return df


def _generate_weather(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate realistic hourly weather for Valencia Jan–Jun 2024."""
    np.random.seed(SEED)
    n = len(dates)

    # Day-of-year for seasonal patterns
    doy = dates.dayofyear.values
    hour = dates.hour.values

    # Monthly mean temps for Valencia
    monthly_mean_temp = {1: 12.0, 2: 13.0, 3: 15.0, 4: 17.0, 5: 20.0, 6: 24.0}
    monthly_amplitude = {1: 5.0, 2: 5.5, 3: 6.0, 4: 7.0, 5: 7.5, 6: 8.0}

    months = dates.month.values
    base_temp = np.array([monthly_mean_temp[m] for m in months])
    amplitude = np.array([monthly_amplitude[m] for m in months])

    # Diurnal cycle: peak at ~15:00, trough at ~5:00
    diurnal = amplitude * np.sin(2 * np.pi * (hour - 5) / 24)
    temp_noise = np.random.normal(0, 1.0, n)
    temperature = base_temp + diurnal + temp_noise

    # Precipitation: two-state model (dry / rainy hour)
    # Probability of rain onset varies by month (more in spring, less in summer)
    monthly_rain_prob = {1: 0.06, 2: 0.06, 3: 0.05, 4: 0.05, 5: 0.04, 6: 0.03}
    monthly_rain_intensity = {1: 2.5, 2: 2.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5}
    rain_probs = np.array([monthly_rain_prob[m] for m in months])
    rain_intensity = np.array([monthly_rain_intensity[m] for m in months])

    # Persistence model: rain states tend to cluster in hours
    precipitation = np.zeros(n)
    rain_state = np.random.random(n) < rain_probs * 0.5
    for i in range(1, n):
        if rain_state[i - 1]:
            rain_state[i] = np.random.random() < 0.7
        else:
            rain_state[i] = np.random.random() < rain_probs[i]

    for i in range(n):
        if rain_state[i]:
            precipitation[i] = np.random.exponential(rain_intensity[i])

    # Wind speed (km/h): mild, with some gusts
    wind_speed = np.random.weibull(2.0, n) * 8 + 4
    wind_speed = np.clip(wind_speed, 0, 45)

    return pd.DataFrame({
        "datetime": dates,
        "temperature_celsius": np.round(temperature, 1),
        "precipitation_mm": np.round(precipitation, 2),
        "wind_speed_kmh": np.round(wind_speed, 1),
    })


def _daily_scale(month: int, station_type: str) -> float:
    return BASE_SCALE[station_type] * MONTHLY_DEMAND[month]


def generate_hourly_demand_csv(stations_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    np.random.seed(SEED)
    random.seed(SEED)

    dates = pd.date_range(START_DATE, END_DATE, freq="h", inclusive="left")
    weather_df = _generate_weather(dates)

    rows = []
    for _, st in stations_df.iterrows():
        sid = st["station_id"]
        sname = st["name"]
        stype = st["type"]
        weekday_profile = np.array(BASE_PROFILES[stype])
        weekend_profile = np.array(WEEKEND_PROFILES[stype])

        for i, dt in enumerate(dates):
            hour = dt.hour
            wday = dt.weekday()  # Mon=0, Sun=6
            is_weekend = 1 if wday >= 5 else 0
            month = dt.month
            date_str = dt.strftime("%Y-%m-%d")

            # ---- base demand ----
            profile = weekend_profile if is_weekend else weekday_profile
            base = profile[hour] * _daily_scale(month, stype)

            # ---- weather effects ----
            temp = weather_df.loc[i, "temperature_celsius"]
            precip = weather_df.loc[i, "precipitation_mm"]
            wind = weather_df.loc[i, "wind_speed_kmh"]

            # Temperature effect: parabolic — optimal around 18-22°C
            temp_effect = np.clip(1.0 - 0.004 * (temp - 20) ** 2, 0.35, 1.15)
            # Rain penalty: strong non-linear
            if precip > 0:
                rain_effect = np.exp(-0.5 * precip)
            else:
                rain_effect = 1.0
            # Wind penalty
            wind_effect = np.clip(1.0 - 0.01 * max(0, wind - 20), 0.6, 1.0)

            weather_factor = temp_effect * rain_effect * wind_effect

            # ---- holiday / special event effects ----
            holiday_factor = 1.0
            if date_str in SPECIAL_DATES:
                holiday_factor = SPECIAL_DATES[date_str][1]

            # ---- assemble demand ----
            demand = base * weather_factor * holiday_factor

            # Poisson-like noise (demand is count data)
            if demand > 0:
                demand = np.random.poisson(max(1, demand))
            else:
                demand = max(0, np.random.poisson(1))

            # Clip to station capacity (max bikes out per hour)
            demand = min(demand, st["capacity"])

            rows.append({
                "station_id": sid,
                "station_name": sname,
                "station_type": stype,
                "ds": dt,
                "y": int(demand),
                "temperature_celsius": temp,
                "precipitation_mm": precip,
                "wind_speed_kmh": wind,
                "hour": hour,
                "day_of_week": wday,
                "is_weekend": is_weekend,
                "month": month,
            })

    df = pd.DataFrame(rows)

    # Sort for readability
    df = df.sort_values(["station_id", "ds"]).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "hourly_demand.csv")
    df.to_csv(path, index=False)
    print(f"[OK] {path} — {len(df):,} rows across {df['station_id'].nunique()} stations")
    print(f"    Date range: {df['ds'].min()} --> {df['ds'].max()}")
    print(f"    Avg demand/station/hour: {df['y'].mean():.2f}")
    return df


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    print("=" * 60)
    print(" Valenbisi Mock Data Generator")
    print(f" Output: {output_dir}")
    print("=" * 60)
    stations_df = generate_stations_csv(output_dir)
    df = generate_hourly_demand_csv(stations_df, output_dir)
    print("=" * 60)
    print(" Done. Ready for app.py.")
    print("=" * 60)


if __name__ == "__main__":
    main()