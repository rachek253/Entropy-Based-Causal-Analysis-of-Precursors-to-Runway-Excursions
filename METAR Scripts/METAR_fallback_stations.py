"""
File Name: METAR_fallback_stations.py

Purpose: This script handles unmatched airports from the NTSB Runway
Excursion dataset by assigning the nearest available ASOS/METAR weather 
station and retrieving weather data for corresponding event dates.

Features: 
- Expands unmatched airports with event date & coordinates
- Finds the nearest weather station using Haversine distance formula
- Fetches weather data with retry/backoff handling (preventing 503 errors)
- Outputs fallback weather dataset for later merging with METAR dataset
"""
import os
import time
import requests
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from io import StringIO

#=================================================================================
load_dotenv()

base_path = os.getenv("PROJECT_DATA_DIRECTORY")
if not base_path:
    raise ValueError("PROJECT_DATA_DIRECTORY is not set. Check your .env file.")

# obtaining necessary file paths
ntsb_path = os.path.join(base_path, "NTSB Runway Excursions.csv")
unmatched_path = os.path.join(base_path, "unmatched_airports.csv")
stations_path = os.path.join(base_path, "asos_stations.csv")

output_path = os.path.join(base_path, "METAR_fallback_stations.csv")
#=================================================================================

#=================================================================================
# ASOS station file standardization
def load_station(path):
    """
    Loads ASOS station metadata and standardizes column names.
    """
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.upper()

    df = df.rename(columns = {
        "CALL": "station",
        "LAT": "lat",
        "LON": "lon"
    })

    required = ["station", "lat", "lon"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        
    df["lat"] = pd.to_numeric(df["lat"], errors = "coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors = "coerce")

    df = df.dropna(subset = ["lat", "lon", "station"])
        
    return df
#=================================================================================

#=================================================================================
# defining Haversine function
def Haversine(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on Earth 
    using the Haversine formula.

    Returns:
        Distance in nautical miles.
    """
    R = 3440.065 # radius of Earth in nautical miles

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 -lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def get_nearest_station(lat, lon, stations_df):
    """
    Finds the nearest weather station to a given lat/lon

    Returns: 
        (station_id, distance_nm)
    """
    stations_df = stations_df.copy()
    stations_df["dist_nm"] = stations_df.apply(lambda row:
                                               Haversine(lat, lon, row["lat"], row["lon"]), axis = 1)
    
    nearest = stations_df.sort_values("dist_nm").iloc[0]

    return nearest["station"], nearest["dist_nm"]
#=================================================================================

#=================================================================================
def fetch_unmatched_weather(station, event_date, max_retries = 4, sleep_time = 2):
    """
    Fetches METAR weather data from Iowa State Mesonet. Handles
    intermittent 503 service errors by retrying with exponential backoff.

    Args:
        station (str): Station identifier
        event_date (datetime): Event date
        max_retries (int): Number of retry attempts
        sleep_time (int): Initial wait time (seconds)

    Returns:
        DataFrame with weather data or empty DataFrame if failed
    """
    if pd.isna(event_date): 
        return pd.DataFrame()
    
    start_date = pd.to_datetime(event_date).date()
    end_date = start_date 

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

    params = {
        "station": station,
        "data" : "tmpf,dwpf,relh,drct,sknt,gust,vsby,alti,p01i,presentwx,metar",
        "year1": start_date.year,
        "month1": start_date.month,
        "day1": start_date.day,
        "year2": end_date.year,
        "month2": end_date.month,
        "day2": end_date.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "report_type": 2
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params = params, timeout = 60)

            if response.status_code == 503:
                raise requests.exceptions.HTTPError("503 Service Unavailable")

            response.raise_for_status()

            text = response.text.strip()
            if text == "" or len(text.splitlines()) < 2:
                return pd.DataFrame()
            
            df = pd.read_csv(StringIO(text), dtype = str)

            if df.empty: 
                return pd.DataFrame()
        
            numeric_cols = ["tmpf", "dwpf", "relh", "drct", 
                        "sknt", "gust", "vsby", "alti", "p01i" ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors = "coerce")

            return df
        except Exception as e:
            print(f"Retry {attempt + 1}/{max_retries} {station} on {event_date}: {e}")
            time.sleep(2**attempt)

    return pd.DataFrame()
#=================================================================================

#=================================================================================
# main pipeline
def main():
    """
    Main execution pipeline:
    1. Load NTSB, ASOS Stations, and unmatched airports datasets
    2. Expand unmatched airports with NTSB data
    3. Find nearest stations 
    4. Fetch data from nearest station
    5. Save results in a .csv
    """
    ntsb = pd.read_csv(ntsb_path)
    unmatched_airports = pd.read_csv(unmatched_path)
    asos_stations = load_station(stations_path)

    ntsb = ntsb.reset_index(drop = True)
    ntsb["NtsbNo"] = ntsb["NtsbNo"].astype(str).str.strip()

    ntsb["AirportID"] = ntsb["AirportID"].astype(str).str.strip().str.upper()
    ntsb["EventDate"] = pd.to_datetime(ntsb["EventDate"], errors = "coerce")

    # merging unmatched stations with NTSB data
    unmatched_expanded = unmatched_airports.merge(
        ntsb[["NtsbNo", "AirportID", "EventDate", "Latitude", "Longitude"]],
        left_on = "UnmatchedAirportID",
        right_on ="AirportID",
        how = "inner"
    )

    unmatched_expanded = unmatched_expanded.drop_duplicates(subset = ["NtsbNo"])

    print(f"Unmatched events to process: {len(unmatched_expanded)}")

    # finding nearest weather station
    nearest_rows = []

    for _, row in unmatched_expanded.iterrows():
        if pd.isna(row["Latitude"]) or pd.isna(row["Longitude"]):
            continue

        station, dist = get_nearest_station(
            row["Latitude"], row["Longitude"], asos_stations
        )
    
        nearest_rows.append({
            "NtsbNo": row["NtsbNo"],
            "AirportID": row["AirportID"],
            "EventDate": row["EventDate"],
            "NearestStation": station,
            "Distance_nm": dist
        })

    nearest_df = pd.DataFrame(nearest_rows).reset_index(drop = True)

    print(f"Nearest stations found: {len(nearest_df)}")

    weather_rows = []
    for _, row in nearest_df.iterrows():
        df = fetch_unmatched_weather(
            row["NearestStation"],
            row["EventDate"]
        )

        if not df.empty:
            df = df.reset_index(drop = True)

            df.loc[:, "NtsbNo"] = row["NtsbNo"]
            df.loc[:, "AirportID"] = row["AirportID"]
            df.loc[:, "EventDate"] = row["EventDate"]
            df.loc[:, "WeatherStation"] = row["NearestStation"]
            df.loc[:, "Distance_nm"] = row["Distance_nm"]

        weather_rows.append(df)

    if weather_rows:
        fallback_station = pd.concat(weather_rows, ignore_index = True)
        fallback_station = fallback_station.reset_index(drop = True)
    else:
        fallback_station = pd.DataFrame()

    fallback_station.to_csv(output_path, index = False)

# running script
if __name__ == "__main__":
    main()