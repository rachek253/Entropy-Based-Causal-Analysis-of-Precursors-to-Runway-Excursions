"""
File Name: METAR.py

Purpose: This module utilizes the 'requests' library to pull archival
hourly weather data for a 24-hour period based for the accident dates
from the NTSB accident data. 
"""
import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv
from io import StringIO

load_dotenv()
# --------------------------------------------------------------------------
# importing NTSB Runway Excursion Dataset
base_path = os.getenv("PROJECT_DATA_DIRECTORY")
if not base_path:
    raise ValueError("PROJECT_DATA_DIRECTORY is not set. Check your .env file.")

csv_path = os.path.join(base_path, "NTSB Runway Excursions.csv")
ntsb = pd.read_csv(csv_path)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# printing the columns and the top of the dataframe
# print(ntsb.head())
# print(ntsb.columns)

# print(ntsb["AirportID"].dropna().head(20))
# print(ntsb["AirportID"].dropna().nunique())
# print(ntsb["AirportID"].dropna().sample(25, random_state=42).tolist())
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# basic data cleanup
ntsb["AirportID"] = ntsb["AirportID"].dropna().astype(str).str.strip().str.upper()
ntsb["EventDate"] = pd.to_datetime(ntsb["EventDate"], utc = True, errors = "coerce")

# removing any missing or junk airport IDS
bad_airports = ["NONE", "N/A", "NA", "NULL", ""]
ntsb = ntsb[~ntsb["AirportID"].isin(bad_airports)].copy()
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# creating a function to test whether a station actually returns weather
def generate_station_candidates(airport_code):
    """
    A function that generates likely METAR station candidates from an
    airport identifier code.
    """
    if pd.isna(airport_code):
        return []
    
    airport_code = str(airport_code).strip().upper()

    # removing junk
    if airport_code in ["NONE", "N/A", "NA", "NULL", ""]:
        return []
    
    METAR_candidates = []
    # checking to see if airport code is in ICAO-format
    if len(airport_code) == 4:
        METAR_candidates.append(airport_code)

    # FAA/IATA format (3-char) to K prefix first
    elif len(airport_code) == 3:
        METAR_candidates.append("K" + airport_code)
        METAR_candidates.append(airport_code)

    return list(dict.fromkeys(METAR_candidates))
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# function to fetch hourly weather summaries from Iowa State MESONET
def fetch_iem_weather(station, event_date, max_retries = 4):
    """
    A function to pull METAR data from Iowa State Mesonet for one station
    and one event date.

    Returns a DataFrame.
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

    print(f"FAILED pulling after retries: {station} on {event_date}")
    return pd.DataFrame()
# --------------------------------------------------------------------------
  
# --------------------------------------------------------------------------
# mapping airports to weather stations

def main():
    unique_airports = ntsb["AirportID"].unique()
    station_map = {}
    unmatched_airports = []

    for airport in unique_airports:
        airport_rows = ntsb[ntsb["AirportID"] == airport]
        airport_rows = airport_rows[airport_rows["EventDate"].notna()]
    
        if airport_rows.empty:
            unmatched_airports.append(airport)
            continue

        test_date = airport_rows.iloc[0]["EventDate"]
        METAR_candidates = generate_station_candidates(airport)

        matched_station = None

        for station in METAR_candidates: 
            test_df = fetch_iem_weather(station, test_date)

            if not test_df.empty:
                matched_station = station
                break
        
        if matched_station:
            station_map[airport] = matched_station
        else:
            unmatched_airports.append(airport)

    print("Matched Airports: ", len(station_map))
    print("Unmatched Airports: ", len(unmatched_airports))

    # saving station maps as csv files
    station_map_df = pd.DataFrame(
        [{"AirportID": k, "WeatherStation": v} for k, v in station_map.items()]
    )
    station_map_df.to_csv(os.path.join(base_path, "airport_station_map.csv"), index = False)

    pd.DataFrame({"UnmatchedAirportID": unmatched_airports}).to_csv(
        os.path.join(base_path, "unmatched_airports.csv"), index = False
    )
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# fetching weather for all events, keyed by airport & event_date
    weather_rows = []

    for _, row in ntsb.iterrows():
        airport = row["AirportID"]
        event_date = row["EventDate"]

        if pd.isna(event_date) or airport not in station_map: 
            continue

        station = station_map[airport]
        weather_df = fetch_iem_weather(station, event_date)

        time.sleep(0.5)

        if not weather_df.empty:
            weather_df["AirportID"] = airport
            weather_df["WeatherStation"] = station
            weather_df["EventDate"] = pd.to_datetime(event_date, utc = True)

            # keeping ntsb row index to merge data 
            weather_df["NtsbNo"] = row["NtsbNo"]

            weather_rows.append(weather_df)


# combining all weather pulls
    if weather_rows:
        all_weather = pd.concat(weather_rows, ignore_index = True)
    else: 
        all_weather = pd.DataFrame()

    weather_output_path = os.path.join(base_path, "Runway_Excursion_Weather.csv")
    all_weather.to_csv(weather_output_path, index = False)

if __name__ == "__main__":
    main()

