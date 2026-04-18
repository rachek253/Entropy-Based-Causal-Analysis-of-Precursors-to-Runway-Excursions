"""
File name: cleaning.py

Purpose: This module implements several cleaning utility functions
to be used on the NTSB & METAR dataset. The functions handle removal of
unnecessary columns, data type standardization, and consistent handling
of missing values to prepare the dataset for analysis and modeling.
"""
import numpy as np
import pandas as pd
#=================================================================================
# dropping missing columns with missing values from dataset or 
# columns with duplicate information
def drop_unnecessary_columns(df):
    """
    Removes irrelevant, duplicate, or mostly empty columns from the
    dataset. This function drops predefined columns that are not useful
    for analysis or contain redundant information. Columns that are 
    entirely missing (all NaN or blank) and columns with excessive missing
    values (>= 95%) are removed from the dataset.

    Returns:
        DataFrame: Cleaned dataset with unnecessary columns removed.
    """
    drop_candidates = [
        "mkey",
        "reportno",
        "hassafetyrec",
        "reporttype",
        "originalpublishdate",
        "eventid",
        "aircraftcategory",
        "scheduled",
        "reportstatus",
        "repgenflag",
        "docketpublishdate",
        "wxcodes",
        "station",
        "airportid_metar",
        "eventdate_metar"
    ]

    to_drop = [col for col in drop_candidates if col in df.columns]
    print("Columns to drop from dataset:", to_drop)

    df = df.drop(columns = to_drop)
    df = df.dropna(axis = 1, how = "all")

    missing_ratio = df.isna().mean()
    df = df.loc[:, missing_ratio < 0.95]
    # print("Remaining columns:", df.columns.tolist())

    return df
#=================================================================================

#=================================================================================
# standardizing data types in each column
def standardize_types(df):
    """
    Standardizes data types safely by keeping identifiers as strings, 
    converts only known numeric columns, and converts datetime columns 
    properly to pandas datetime (UTC). Only explicilty listed numeric
    columns are converted to avoid accidentally transforming accident
    identifiers (e.g., ntsbno, airportid).

    Returns:   
        DataFrame: Dataset with standardized datatypes.  
    """
    datetime_cols = [c for c in df.columns if "date" in c.lower() or "valid" in c.lower()]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors = "coerce", utc = True)

    # converting explicit numeric columns to numeric
    numeric_cols = [
        "fatalinjurycount",
        "seriousinjurycount",
        "minorinjurycount", 
        "latitude",
        "longitude",
        "numberofengines",
        "tmpf",
        "dwpf",
        "relh",
        "drct", 
        "sknt",
        "gust",
        "vsby",
        "alti",
        "p01i",
        "distance_nm"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors = "coerce")

    # converting explicit string columns to string
    string_cols = [
        "ntsbno",
        "eventtype",
        "city",
        "state",
        "country",
        "n",
        "highestinjurylevel",
        "probablecause",
        "make",
        "model",
        "airportname",
        "purposeofflight",
        "far",
        "aircraftdamage",
        "weathercondition",
        "operator",
        "docketurl",
        "metar",
        "weather_source"
    ]

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df
#=================================================================================

#=================================================================================
# filling missing values
def fill_missing_values(df):
    """
    Fills missing values for key columns with appropriate defaults.
    Missing values in the NTSB dataset mainly occur when the NTSB is 
    not the primary investigator in an accident, usually when a 
    US-registered aircraft is involved in an accident in another 
    country. These accidents are still of value to the analysis,
    so missing values must be standardized. 

    Rules:
        - probablecause set to "UNKNOWN"
        - weathercondition set to "UNKNOWN"
        - purposeofflight set to "UNKNOWN" if missing
        - docketurl set to "NA"
        - state "NA" for non-US entries if required
        - operator set to "NA" if missing
        - aircraftdamage set to "NA"
        - FAR operations set to "NA"
        - 

    Returns:
        DataFrame with filled values
    """
    # confirming all columns are lowercase for consistent formatting
    df.columns = df.columns.str.strip()

    if "probablecause" in df.columns:
        df["probablecause"] = (df["probablecause"].fillna("UNKNOWN").replace("", "UNKNOWN"))

    if "weathercondition" in df.columns:
        df["weathercondition"] = df["weathercondition"].str.upper()
        df["weathercondition"] = (df["weathercondition"].fillna("UNKNOWN").replace(["", "Unknown", "UNKNOWN", "unknown"], "UNKNOWN"))

    if "purposeofflight" in df.columns:
        df["purposeofflight"] = (df["purposeofflight"].fillna("UNKNOWN").replace("", "UNKNOWN"))

    if "docketurl" in df.columns:
        df["docketurl"] = (df["docketurl"].fillna("NA").replace("", "NA"))

    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
        # replacing missing/blank values
        df["state"] = (df["state"].fillna("NA").replace("", "NA"))

    if "operator" in df.columns:
        df["operator"] = (df["operator"].fillna("NA").replace("", "NA"))

    if "aircraftdamage" in df.columns:
        df["aircraftdamage"] = (df["aircraftdamage"].fillna("NA").replace("", "NA"))

    if "far" in df.columns:
        df["far"] = (df["far"].fillna("NA").replace("", "NA"))

    # preserving 'None' as an injury type and not python None
    if "highestinjurylevel" in df.columns:
        df["highestinjurylevel"] = (
            df["highestinjurylevel"].fillna("None").replace("", "None").astype("string")
        )

    # dealing with numerical weather columns
    weather_numeric_cols = [
        "tmpf", "dwpf", "relh", "drct",
        "sknt", "gust", "vsby", "alti", "p01i"
    ]

    for col in weather_numeric_cols:
        if col in df.columns:
            if col == "p01i":
                # precipitation assuming 0 if missing
                df[col] = df[col].fillna(0)
            else:
                if "weatherstation" in df.columns:
                    # station-level median to replace missing values
                    df[col] = df.groupby("weatherstation")[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                df[col] = df[col].fillna(df[col].median())

    return df
#=================================================================================

#=================================================================================
# adding crosswind and tail components to data
def add_wind_components(df):
    """
    Computes crosswind and tailwind components from windspeed and 
    direction. This function approximates runway-relative wind effects 
    assuming runway orientation is unknown. 

    Returns: 
        DataFrame: Updated dataset with crosswind and tailwind components
    """
    if "sknt" not in df.columns or "drct" not in df.columns:
        return df
    
    # assume runway orientation is unknown 
    runway_heading = 0
    wind_dir_rad = np.deg2rad(df["drct"])
    runway_rad = np.deg2rad(runway_heading)

    wind_speed = df["sknt"]

    # vector decomposition
    df["crosswind_component"] = wind_speed * np.sin(wind_dir_rad - runway_rad)
    df["tailwind_component"] = wind_speed * np.cos(wind_dir_rad - runway_heading)

    return df