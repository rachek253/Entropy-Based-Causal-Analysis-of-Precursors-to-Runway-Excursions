"""
File Name: METAR_event_aligned_merge.py

Purpose: This script creates a time-aligned weather dataset by merging
NTSB runway excursion events with METAR weather observations. It replaces
any truncated or inconsistent EventDate values in the combined weather
dataset with the official timestamp from the NTSB dataset and selects the 
closest available hourly METAR observation for each accident event. 
"""
import os
import pandas as pd

from dotenv import load_dotenv

#=================================================================================
load_dotenv()

base_path = os.getenv("PROJECT_DATA_DIRECTORY")
if not base_path: 
    raise ValueError("PROJECT_DATA_DIRECTORY is not set. Check your .env file.")

ntsb_path = os.path.join(base_path, "NTSB Runway Excursions.csv")
combined_weather_path = os.path.join(base_path, "METAR_Weather_Combined.csv")

output_path = os.path.join(base_path, "METAR_TimeAligned_Final.csv")
#=================================================================================

#=================================================================================
def load_datasets():
    """
    Loads NTSB and combined METAR datasets and standardizes timestamps.

    Returns:
        ntsb (DataFrame): NTSB dataset with EventDate timestamps
        metar (DataFrame): combined METAR dataset
    """
    ntsb = pd.read_csv(ntsb_path)
    metar = pd.read_csv(combined_weather_path)

    ntsb["EventDate"] = pd.to_datetime(ntsb["EventDate"], errors = "coerce", utc = True)

    # converting METAR observation time
    metar["valid"] = pd.to_datetime(metar["valid"], errors = "coerce", utc = True)

    if "NtsbNo" not in ntsb.columns:
        ntsb = ntsb.reset_index().rename(columns = {"index": "NtsbNo"})

    return ntsb, metar
#=================================================================================

#=================================================================================
# splitting datetime into date + time
def split_datetime(df):
    """
    Splits datetime columns into separate date and time columns
    """
    df["valid_date"] = df["valid"].dt.date
    df["valid_time"] = df["valid"].dt.time

    df["event_date"] = df["EventDate"].dt.date
    df["event_time"] = df["EventDate"].dt.time

    return df

#=================================================================================
# aligning METAR observations to closest event time
def closest_weather_to_event(ntsb, metar):
    """
    Matches each accident (NtsbNo) with closest METAR observation in time.

    Steps:
        1. Merge weather & NTSB on NtsbNo
        2. Compute absolute time difference
        3. Select closest observation per accident

    Returns:
        aligned_df (DataFrame): One row per accident with closest weather.
    """
    ntsb_time = ntsb["EventDate"]
    ntsb_id = ntsb["NtsbNo"]

    subset = metar[metar["valid"].notna()].copy()

    if subset.empty:
        return None

    subset["time_diff"] = (subset["valid"] - ntsb_time).abs()

    closest_weather = subset.loc[subset["time_diff"].idxmin()].copy()

    closest_weather["NtsbNo"] = ntsb_id
    closest_weather["EventDate"] = ntsb_time

    return closest_weather
#=================================================================================

#=================================================================================
# aligning weather to accident events
def align_all_events(ntsb, metar):
    """
    Aligns all NTSB events to closest METAR observations.

    Returns:
        DataFrame with exactly 1 row per event (where possible)
    """
    results = []

    for _, event in ntsb.iterrows():
        if pd.isna(event["EventDate"]):
            continue

        match = closest_weather_to_event(event, metar)

        if match is not None:
            results.append(match)

    aligned = pd.DataFrame(results)
    aligned = aligned.loc[:, ~aligned.columns.duplicated()]

    # sorting from most recent accident to oldest to match NTSB format
    aligned = aligned.sort_values("EventDate", ascending = False).reset_index(drop = True)
    
    return aligned
#=================================================================================

#=================================================================================
def main():
    ntsb, metar = load_datasets()

    aligned = align_all_events(ntsb, metar)

    print(f"Aligned rows: {len(aligned)}")

    aligned.to_csv(output_path, index = False)


if __name__ == "__main__":
    main()