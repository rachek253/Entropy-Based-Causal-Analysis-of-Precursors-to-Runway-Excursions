"""
File Name: eda.py

Purpose: This module performs exploratory analysis (EDA) on the
finalized NTSB & METAR dataset. It provides an overview of the dataset,
a missing value analysis, summary statistics for numerical variables,
and distribution insights for key variables.
"""
import os
import pandas as pd

from scipy.stats import chi2_contingency, ttest_ind
#=================================================================================

#=================================================================================
# overview of the dataset
def dataset_overview(df, label = "Dataset"):
    """
    Prints high-level dataset information. 

    Returns:
        - Number of rows and columns in the dataset
        - Column names
        - data types
    """
    print(f"\n======= {label} Overview =======")
    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)
#=================================================================================

#=================================================================================
# handling missing values
def missing_value_summary(df):
    """
    Computes missing value statistics for each column.

    Returns:
        DataFrame showing both missing count and missing percentage
    """
    missing_data = df.isna().sum()
    percent = (missing_data / len(df)) * 100


    summary = pd.DataFrame({
        "missing_count": missing_data,
        "missing_percent": percent
    }).sort_values(by = "missing_percent", ascending = False)

    print("\n======= Missing Value Summary =======")
    print(summary.head(20))

    return summary
#=================================================================================

#=================================================================================
def numeric_summary(df):
    """
    Provides descriptive statistics for numerical columns.

    Returns:
        DataFrame with summary statistics (mean, std, min, max, etc.)
    """
    numeric_df = df.select_dtypes(include = ["number"])
    print("\n======= Numeric Column Summary =======")
    summary = numeric_df.describe()

    latex_table = summary.to_latex(
        float_format = "%.3f",
        longtable = True,
        escape = False,
        caption = "Summary Statistics for Numeric Variables",
        label = "tab: numeric_summary"
    )

    print(latex_table)
#=================================================================================

#=================================================================================
# categorical summary for important variables
def categorical_summary(df, top_n = 10): 
    """
    Provides basic frequency counts for key categorical variables. Key 
    categorical columns include important information about runway excursion
    events, including highest injury counts, event types (Accident vs Incident), 
    weather condition, damage to aircraft, purpose of flight, FAR number
    (category of aircraft operations), and weather_source (Primary METAR
    vs Fallback METAR station).

    Returns:
        Prints value counts for important categorical columns.
    """
    categorical_cols = [
        "eventtype",
        "highestinjurylevel",
        "purposeofflight",
        "far",
        "aircraftdamage",
        "weathercondition",
        "weather_station",
        "make",
        "model"
    ]

    print("\n======= Categorical Data Summary =======")

    for col in categorical_cols:
        if col in df.columns:
            print(f"\nTop {top_n} values for '{col}':")

            cleaned = (df[col].astype(str)
                       .str.strip()
                       .str.upper())
            
            print(cleaned.value_counts().head(top_n))
#=================================================================================

#=================================================================================
# aircraft make and model insights
def aircraft_insights(df):
    """
    Provides a deeper insight into aircraft make and model combinations
    within the accident dataset.

    Returns:
        - Most common aircraft makes
        - Most common models
        - Top make-model combinations
    """
    if "make" in df.columns and "model" in df.columns:
        print("\n======= Aircraft Insights =======")
        df = df.copy()
        df["make_clean"] = df["make"].astype(str).str.strip().str.upper()
        df["model_clean"] = df["model"].astype(str).str.strip().str.upper()

        print("\nTop Aircraft Makes:")
        print(df["make_clean"].value_counts().head(10))

        print("\nTop Aircraft Models:")
        print(df["model_clean"].value_counts().head(10))

        print("\nTop Make & Model Combinations:")
        combo = df["make_clean"] + " " + df["model_clean"]
        print(combo.value_counts().head(10))
#=================================================================================

#=================================================================================
# distribution of weather sources
def weather_source_summary(df):
    """
    Summarizes distribution of weather data sources. METAR information
    obtained using the airport identifier is marked 'PRIMARY', and METAR
    information obtained using Haversine distance formula for closest 
    weather station based on longitude and latitude of an airport is
    marked 'FALLBACK'. 

    Returns:
        - Summary of weather data sources
    """
    if "weather_source" in df.columns:
        print("\n======= Weather Source Distribution =======")
        print(df["weather_source"].describe())
        print(df["weather_source"].value_counts(normalize = True))

#=================================================================================
# computed time-alignment evaluation
def time_alignment_quality(df):
    """
    Evaluates how close METAR observations match up to accident event
    time and prints the results for observations within 30 minutes and 
    within one hour.

    Returns:
        - Mean number of observations matching up within 1 hour
        - Mean number of observation matching up within 30 minutes
    """
    df["time_diff"] = pd.to_timedelta(df["time_diff"], errors="coerce")

    if "time_diff" in df.columns:
        print("\n======= Time Alignment Quality =======")
        print(df["time_diff"].describe())

        within_1hr = (df["time_diff"] <= pd.Timedelta(hours = 1)).mean()
        within_30min = (df["time_diff"] <= pd.Timedelta(hours = 0.5)).mean()

        print(f"Within 1 hour: {within_1hr:.2%}")
        print(f"Within 30 minutes: {within_30min:.2%}")
#=================================================================================

#=================================================================================
# runway excursion outcome severity
def outcome_severity(df):
    """
    Provides summaries for accident outcomes with information about
    the highest injury level (None, Minor, Serious, and Fatal) and 
    damage to aircraft (NA, None, Minor, Substantial, Destroyed).

    Returns: 
        Summary of highest injury level distribution
        Summary of damage to aircraft distribution
    """
    print("\n======= Outcome Severity =======")

    if "highestinjurylevel" in df.columns:
        print(df["highestinjurylevel"].describe())
        print(df["highestinjurylevel"].value_counts(normalize = True))


    if "aircraftdamage" in df.columns:
        print(df["aircraftdamage"].describe())
        print(df["aircraftdamage"].value_counts(normalize = True))
#=================================================================================

#=================================================================================
# geographical summary of accidents by state
def geographical_summary(df): 
    """
    Summarizes the geographical distribution of NTSB events in the dataset.

    Returns:
        Summary of states involved in events.
    """
    if "state" in df.columns:
        print("\n======= Frequency Count for States =======")
        print(df["state"].value_counts().head(20))
#=================================================================================

#=================================================================================
# summarizing events by operation type
def operation_summary(df):
    """
    Provides summaries for types of operations involved in events. Federal
    Aviation Regulations (FAR) 91, 121, and 135 define operation types. 
    Part 91 airports are considered non-commerical or private, Part 121
    airports are for scheduled commercial aircarriers, and Part 135 airports
    are charter/on-demand. 

    Returns:
        Summary of FAR operations
        Summary of purpose of flight
    """
    print("\n======= Operation Summary ========")
    if "far" in df.columns: 
        print(df["far"].describe())
        print(df["far"].value_counts())


    if "purposeofflight" in df.columns:
        print(df["purposeofflight"].describe())
        print(df["purposeofflight"].value_counts())
#=================================================================================

#=================================================================================
# weather condition breakdown
def weather_condition_summary(df):
    """
    Summarizes reported weather conditions. Aviation weather safety limits
    are defined as Visual Meteorlogical Conditions (VMC) or Instrument
    Meteorological Conditions (IMC), and determine how pilots fly the
    aircraft. VMC is clear weather conditions that allow visual navigation
    and IMC represents conditions below minimums for visual flight (i.e. 
    fog, low clouds, heavy rain, etc.) requiring reliance on instruments.

    Returns:
        - Counts of VMC, IMC or UNKNOWN weather conditions.
    """
    if "weathercondition" in df.columns:
        print("\n======= Weather Conditions =======")
        print(df["weathercondition"].describe())
        print(df["weathercondition"].value_counts())
#=================================================================================

#=================================================================================
# chi-square test 
def run_chi_square(df, col1, col2):
    """
    Performs chi-square test of independence between two categorical
    variables. Tests whether two categorical variables are statistically
    dependent.

    Returns:
        chi2_stat, p_value, degrees_of_freedom
    """ 
    if col1 not in df.columns or col2 not in df.columns:
        return None
    
    contingency_table = pd.crosstab(df[col1], df[col2])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\n======= Chi-Square Test =======")
    print(f"Variables: {col1} vs {col2}")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.6f}")
    print(f"Degrees of Freedom: {dof}")

    if p < 0.05: 
        print("Result: Statistically significant relationship (reject H0)")
    else:
        print("Result: No strong evidence of association (fail to reject H0)")

    return chi2, p, dof
#=================================================================================

#=================================================================================
# T-test numeric group comparison
def run_ttest(df, numeric_col, group_col, group_a, group_b): 
    """
    Performs independent t-test between two groups. Determines whether
    the means of a numeric variable differ significantly between two
    categorical groups.

    Example:
        - visibility (vsby) between IMC vs VMC
        - wind speed between accident severity groups

    Returns:
        t_stat, p_value
    """
    if numeric_col not in df.columns or group_col not in df.columns:
        return None
    
    group1 = df[df[group_col] == group_a][numeric_col].dropna()
    group2 = df[df[group_col] == group_b][numeric_col].dropna()

    t_stat, p = ttest_ind(group1, group2, equal_var = False)

    print("\n======= T-Test =======")
    print(f"{numeric_col} | {group_a} vs {group_b}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p:.6f}")

    if p < 0.05:
        print("Result: Significant difference between groups")
    else:
        print("Result: No significant difference")

    return t_stat, p
