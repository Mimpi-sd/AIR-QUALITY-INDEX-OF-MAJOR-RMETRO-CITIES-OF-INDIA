"""
Download daily air quality (CAMS model via Open-Meteo) for several Indian cities.

Usage:
    pip install requests pandas
    python download_city_air_quality.py                      # defaults below
    python download_city_air_quality.py --end 2026-10-12     # extend to a later date

Output (in ./city_air_quality/):
    <city>_air_quality_daily.csv   one file per city, one row per day
    all_cities_daily.csv           the same data stacked, with a 'city' column

Notes
- Source: Copernicus Atmosphere Monitoring Service (CAMS) through Open-Meteo.
  Please credit "CAMS ENSEMBLE data provider" and "Open-Meteo" (CC BY 4.0).
- Values are MODEL output for a grid cell (not ground-station readings).
- Hourly values are averaged to daily means (Asia/Kolkata calendar day).
  Bengaluru is downloaded again so that every city is aggregated identically.
- Defaults match your Bengaluru file: 2022-08-04 to 2026-02-18.
"""
import argparse
import os
import time

import pandas as pd
import requests

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

CITIES = {
    # name: (latitude, longitude)
    "bengaluru": (12.972, 77.594),
    "hyderabad": (17.385, 78.4867),
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "kolkata": (22.5726, 88.3639),
}

VARS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
        "aerosol_optical_depth", "dust", "uv_index", "us_aqi", "european_aqi"]


def fetch_chunk(lat, lon, start, end, tries=5):
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(VARS),
              "start_date": start, "end_date": end, "timezone": "Asia/Kolkata"}
    for k in range(tries):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            h = r.json()["hourly"]
            return pd.DataFrame(h)
        if r.status_code == 429:           # rate limit: wait and retry
            time.sleep(20 * (k + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {tries} tries: {start}..{end}")


def download_city(name, lat, lon, start, end):
    frames = []
    # request in 6-month chunks to stay well within API limits
    s, last = pd.Timestamp(start), pd.Timestamp(end)
    while s <= last:
        e = min(s + pd.DateOffset(months=6) - pd.Timedelta(days=1), last)
        frames.append(fetch_chunk(lat, lon, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        s = e + pd.Timedelta(days=1)
        time.sleep(1)
    h = pd.concat(frames).drop_duplicates("time")
    h["time"] = pd.to_datetime(h["time"])
    d = h.set_index("time").resample("D").mean()          # daily mean of hourly values
    d.index.name = "date"
    return d.round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-08-04")
    ap.add_argument("--end", default="2026-02-18")
    ap.add_argument("--out", default="city_air_quality")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allc = []
    for name, (lat, lon) in CITIES.items():
        print("downloading", name, flush=True)
        d = download_city(name, lat, lon, a.start, a.end)
        d.to_csv(f"{a.out}/{name}_air_quality_daily.csv")
        allc.append(d.reset_index().assign(city=name))
        print(f"  {name}: {len(d)} days, {d['pm2_5'].notna().sum()} with PM2.5")
    pd.concat(allc).to_csv(f"{a.out}/all_cities_daily.csv", index=False)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
