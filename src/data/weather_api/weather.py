import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import pandas as pd

# Load .env file
load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise EnvironmentError("Missing WEATHER_API_KEY in environment variables.")

base_url = "http://api.weatherapi.com/v1/history.json"
date = "2025-05-17"  # or use dynamic date

locations = {
    "Downtown": "Houston",
    "Cypress": "Cypress,TX",
    "Atascocita": "Atascocita,TX",
    "Pasadena": "Pasadena,TX",
    "Bellaire": "Bellaire,TX"
}

dfs = []

for name, city in locations.items():
    params = {
        "key": API_KEY,
        "q": city,
        "dt": date
    }

    try:
        r = requests.get(base_url, params=params)
        if r.status_code == 200:
            hourly = r.json()["forecast"]["forecastday"][0]["hour"]
            df = pd.DataFrame({
                "time": [entry["time"] for entry in hourly],
                name: [entry["temp_c"] for entry in hourly]
            })
            dfs.append(df)
        else:
            print(f"Failed for {name}: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Error for {name}: {e}")

# Combine and compute averages
if not dfs:
    raise ValueError("No weather data was collected.")

weather_df = dfs[0]
for df in dfs[1:]:
    weather_df = weather_df.merge(df, on="time", how="outer")

weather_df.dropna(inplace=True)
weather_df["avg_temperature"] = weather_df[list(locations.keys())].mean(axis=1)

# Add Fahrenheit column
weather_df["avg_temperature_f"] = weather_df["avg_temperature"] * 9 / 5 + 32

weather_df = weather_df.sort_values("time")
print(weather_df[["time", "avg_temperature", "avg_temperature_f"]].head())
