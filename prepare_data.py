"""Create the analysis-ready daily files data/analysis/<city>/air_quality_historical.csv for all seven cities.

CAMS cities: copied unchanged (the analysis drops the first three days that have no PM2.5).
Delhi: the converted station series is cut to 2023-01-01 .. 2026-06-30 (a 31-day gap starts on 2026-07-01)
and isolated gaps of at most 7 days are filled by linear interpolation (10 days in total).
Run from the repository root:  python src/prepare_data.py
"""
import os, shutil
import pandas as pd

for c in ["bengaluru", "chennai", "hyderabad", "ahmedabad", "mumbai", "kolkata"]:
    os.makedirs(f"data/analysis/{c}", exist_ok=True)
    shutil.copy(f"data/cams/{c}/air_quality_historical.csv", f"data/analysis/{c}/air_quality_historical.csv")

cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
        "aerosol_optical_depth", "dust", "uv_index", "us_aqi", "european_aqi"]
d = (pd.read_csv("data/delhi_processed/delhi_air_quality_daily.csv", parse_dates=["date"]).set_index("date")
     .loc["2023-01-01":"2026-06-30"].asfreq("D"))
d = d[cols].interpolate(limit=7, limit_area="inside")
os.makedirs("data/analysis/delhi", exist_ok=True)
d.reset_index().to_csv("data/analysis/delhi/air_quality_historical.csv", index=False)
print("prepared 7 cities; Delhi rows:", len(d), "PM2.5 missing:", int(d.pm2_5.isna().sum()))
