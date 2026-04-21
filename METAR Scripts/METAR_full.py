"""
File Name: METAR_full.py

Purpose: Combines primary METAR weather dataset with fallback (next
nearest) station dataset. Standardizes columns and prepares for matching
with NTSB events. 
"""
import os
import pandas as pd

from dotenv import load_dotenv

#=================================================================================
load_dotenv()

base_path = os.getenv("PROJECT_DATA_DIRECTORY")
if not base_path: 
    raise ValueError("PROJECT_DATA_DIRECTORY is not set. Check your .env file.")

primary_weather_path = os.path.join(base_path, "Runway_Excursion_Weather.csv")
fallback_weather_path = os.path.join(base_path, "METAR_fallback_stations.csv")

output_path = os.path.join(base_path, "METAR_Weather_Combined.csv")
#=================================================================================

#=================================================================================
# loading and standardizing datasets
def load_standardize_datasets():
    """
    Loads both weather datasets and standardizes column names.
    """
    primary_metar = pd.read_csv(primary_weather_path)
    fallback_metar = pd.read_csv(fallback_weather_path)

    if "EventDate" in primary_metar.columns:
        primary_metar["EventDate"] = pd.to_datetime(primary_metar["EventDate"], errors = "coerce")

    if "EventDate" in fallback_metar.columns:
        fallback_metar["EventDate"] = pd.to_datetime(fallback_metar["EventDate"], errors = "coerce")

    primary_metar["Weather_Source"] = "PRIMARY"
    fallback_metar["Weather_Source"] = "FALLBACK"

    # adding nautical mile distance in primary; setting to 0.0 as it uses airport station
    if "Distance_nm" not in primary_metar.columns:
        primary_metar["Distance_nm"] = 0.0

    return primary_metar, fallback_metar
#=================================================================================

#=================================================================================
# combining weather datasets and sorting by accident date
def combine_weather(primary_metar, fallback_metar): 
    """
    Combines both weather datasets and sorts chronologically by event date.
    """
    combined = pd.concat([primary_metar, fallback_metar], ignore_index = True)

    combined["EventDate"] = pd.to_datetime(combined["EventDate"], errors = "coerce", utc = True)
    combined["valid"] = pd.to_datetime(combined["valid"], errors = "coerce", utc = True)    

    # removing any duplicate columns
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # sorting weather info by time
    combined = combined.sort_values(
        by = ["EventDate", "valid"],
        ascending = [False, True]
        ).reset_index(drop = True)

    return combined
#=================================================================================

#=================================================================================
# main file 
def main():
    primary_metar, fallback_metar = load_standardize_datasets()

    combined = combine_weather(primary_metar, fallback_metar)

    # showing length of each dataset
    print(f"Primary METAR weather rows: {len(primary_metar)}")
    print(f"Fallback METAR rows: {len(fallback_metar)}")
    print(f"Total combined weather rows: {len(combined)}")

    combined.to_csv(output_path, index = False)
#=================================================================================

#=================================================================================
if __name__ == "__main__":
    main()