"""
Convert Delhi CPCB/DPCC hourly station data into the same daily, one-row-per-day format
as the CAMS city files (Bengaluru, Mumbai, ...).

python convert_delhi.py <delhi_folder> <output_folder>

Rules (all documented in README_delhi_conversion.md that this script also writes):
 * date = local calendar day (Asia/Kolkata), taken from the timestamp as written.
 * hourly ozone above 500 ug/m3 (one reading of 932) set to missing; nothing else is edited.
 * a station-day needs >= 16 valid hourly values (75 %, the CPCB rule for a 24-h average).
 * city-day value = mean over stations with a valid station-day.
 * CO converted from mg/m3 to ug/m3 (x1000) to match the CAMS files.
"""
import sys, os
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

c = pd.read_csv(f"{src}/caaqms_hourly.csv", low_memory=False)
st = pd.read_csv(f"{src}/stations.csv").set_index("station_id").station_name
c["station"] = c.station_id.map(st)
c["date"] = pd.to_datetime(c.timestamp.str.slice(0, 10))
c["o3"] = c["o3"].where(c["o3"] <= 500)
c["co"] = c["co"] * 1000.0

MAP = {"pm25": "pm2_5", "pm10": "pm10", "co": "carbon_monoxide", "no2": "nitrogen_dioxide",
       "so2": "sulphur_dioxide", "o3": "ozone"}
WX = {"temperature": "temperature_c", "humidity": "humidity_pct", "wind_speed": "wind_speed_ms",
      "solar_radiation": "solar_radiation_wm2"}
NEED = 16

def station_day(col):
    g = c.groupby(["station", "date"])[col]
    d = pd.DataFrame({"mean": g.mean(), "n": g.count()})
    return d["mean"].where(d["n"] >= NEED).unstack("station")   # date x station

sd = {k: station_day(k) for k in list(MAP) + list(WX)}

# rainfall: daily total, only when >= 16 hourly readings exist (coverage is poor, so mostly missing)
g = c.groupby(["station", "date"]).rainfall
rt = pd.DataFrame({"s": g.sum(min_count=1), "n": g.count()})
sd["rainfall"] = rt["s"].where(rt["n"] >= NEED).unstack("station")

idx = pd.date_range(c.date.min(), c.date.max(), freq="D")
city = pd.DataFrame(index=idx)
for k, name in MAP.items():
    city[name] = sd[k].mean(axis=1).reindex(idx)
city["n_stations_pm25"] = sd["pm25"].notna().sum(axis=1).reindex(idx).fillna(0).astype(int)

CORE = ["Anand Vihar", "Bawana", "ITO", "Jahangirpuri", "Punjabi Bagh", "R K Puram"]   # run 2024-01 to 2026-08
core = sd["pm25"][CORE]
city["pm2_5_core6"] = core.mean(axis=1).where(core.notna().sum(axis=1) >= 4).reindex(idx)   # >=4 of 6 stations
city["pm2_5_dwarka"] = sd["pm25"]["Dwarka-Sector 8"].reindex(idx)      # single-station series 2023-2025
for k, name in WX.items():
    city[name] = sd[k].mean(axis=1).reindex(idx)
city["rainfall_mm"] = sd["rainfall"].mean(axis=1).reindex(idx)

# columns that exist in the CAMS files but have no station equivalent (left empty, not guessed)
for col in ["aerosol_optical_depth", "dust", "uv_index", "us_aqi", "european_aqi"]:
    city[col] = np.nan
cams_cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
             "aerosol_optical_depth", "dust", "uv_index", "us_aqi", "european_aqi"]
extra = ["n_stations_pm25", "pm2_5_core6", "pm2_5_dwarka", "temperature_c", "humidity_pct", "wind_speed_ms",
         "solar_radiation_wm2", "rainfall_mm"]
city = city[cams_cols + extra].round(3)
city.index.name = "date"
city["source"] = "CPCB/DPCC ground stations (measured)"

city.to_csv(f"{out}/delhi_air_quality_daily.csv")

# same date index as the other six files (2022-08-01 .. 2026-02-18) so it stacks directly
al = city.reindex(pd.date_range("2022-08-01", "2026-02-18", freq="D"))
al.index.name = "date"
al["source"] = "CPCB/DPCC ground stations (measured)"
al.to_csv(f"{out}/delhi_air_quality_daily_aligned_to_other_cities.csv")

# station-level daily PM2.5 (long format) for anyone who wants to do station-fixed analysis
sd["pm25"].round(2).rename_axis("date").to_csv(f"{out}/delhi_station_daily_pm2_5.csv")
print(city.shape, city.index.min().date(), city.index.max().date())
print("valid PM2.5 days:", int(city.pm2_5.notna().sum()), "| aligned file rows:", len(al))
