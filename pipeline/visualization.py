"""
File Name: visualization.py

Purpose: This module provides visualization tools for exploring the 
NTSB & METAR dataset both before and after cleaning. It generates plots
to further understanding of distributions, relationship between variables,
as well as key avaition safety insights.

This module focuses on:
    - Comparing raw vs cleaned data
    - Exploring injury severity and aircraft damage
    - Understanding weather impacts on accidents
    - Analyzing temporal and geographical patterns
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#=================================================================================

#=================================================================================
# function to save figures as .svg files
def save_figure(filename, folder = "figures"):
    """
    Saves the current matplotlib figure as an SVG file.

    Returns: 
        'figures' folder full of matplot figures.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, f"{filename}.svg")
    plt.tight_layout()
    plt.savefig(filepath, format = "svg")

#=================================================================================
def plot_missing_values(df, title = "Missing Values", save = False):
    """
    Visualizes missing data percentages for each column. Identifies
    remaining gaps after cleaning.
    """
    missing_percent = (df.isna().mean() * 100).sort_values(ascending = False)

    plt.figure()
    missing_percent.head(20).plot(kind = "bar")
    plt.title(title)
    plt.ylabel("Missing (%)")
    plt.xlabel("Columns")
    plt.xticks(rotation = 45)

    if save:
        save_figure("missing_values")

    plt.show()
#=================================================================================

#=================================================================================
# plotting injury severity distribution
def plot_injury_severity(df, save = False):
    """
    Plots distribution of injury severity levels. Shows the severity of 
    runway excursion events.
    """
    if "highestinjurylevel" not in df.columns:
        return
    
    counts = df["highestinjurylevel"].astype(str).value_counts()

    plt.figure()
    counts.plot(kind = "bar")
    plt.title("Injury Severity Distribution")
    plt.xlabel("Injury Level")
    plt.ylabel("Count")
    plt.xticks(rotation = 45)

    if save:
        save_figure("injury_severity_distribution")

    plt.show()
#=================================================================================

#=================================================================================
# aircraft damange distribution
def plot_aircraft_damage(df, save = False):
    """
    Plots aircraft damage distribution. Indicates severity of physical 
    damage to aircraft in events.
    """
    if "aircraftdamage" not in df.columns:
        return
    
    counts = df["aircraftdamage"].astype(str).value_counts()

    plt.figure()
    counts.plot(kind = "bar")
    plt.title("Aircraft Damage Distribution")
    plt.xlabel("Damage Level")
    plt.ylabel("Count")
    plt.xticks(rotation = 45)

    if save:
        save_figure("Aircraft Damage Distribution")

    plt.show()
#=================================================================================

#=================================================================================
# weather condition distribution
def plot_weather_conditions(df, save = False):
    """
    Visualizes weather condition categories.
    """
    if "weathercondition" not in df.columns:
        return
    
    counts = df["weathercondition"].astype(str).value_counts()

    plt.figure()
    counts.plot(kind = "bar")
    plt.title("Weather Condition Distribution")
    plt.xlabel("Condition")
    plt.ylabel("Count")
    plt.xticks(rotation = 45)

    if save: 
        save_figure("weather_condition_distribution")

    plt.show()
#=================================================================================

#=================================================================================
# temperature distribution
def plot_temperature_distribution(df, save = False):
    """
    Plots temperature distributions (tmpf) in degrees F.
    """
    if "tmpf" not in df.columns:
        return
    
    plt.figure()
    df["tmpf"].dropna().hist(bins = 30)
    plt.title("Temperature Distribution (F)")
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")

    if save: 
        save_figure("temperature_distribution")

    plt.show()
#=================================================================================

#=================================================================================
# wind speed distribution
def plot_wind_speed(df, save = False):
    """
    Plots wind speed distribution (sknt).
    """
    if "sknt" not in df.columns:
        return
    
    plt.figure()
    df["sknt"].dropna().hist(bins = 30)
    plt.title("wind Speed Distributions (knots)")
    plt.xlabel("Wind Speed")
    plt.ylabel("Frequency")

    if save:
        save_figure("wind_speed_distribution")

    plt.show()
#=================================================================================

#=================================================================================
# plotting visibility vs injury severity
def plot_visibility_vs_severity(df, save = False):
    """
    Examines relationship between visibility and injury severity.
    """
    if "vsby" not in df.columns or "highestinjurylevel" not in df.columns:
        return
    
    grouped = df.groupby("highestinjurylevel")["vsby"].mean()

    plt.figure()
    grouped.plot(kind = "bar")
    plt.title("Average Visibility by Injury Severity")
    plt.xlabel("Injury Level")
    plt.ylabel("Average Visibility")
    plt.xticks(rotation = 45)

    if save:
        save_figure("avg_visibiliy_vs_severity")

    plt.show()
#=================================================================================

#=================================================================================
# comparing IMC vs VMC risk comparison
def plot_imc_vs_vmc_risk(df, save = False):
    """
    Compares severity risk between IMC and VMC conditions. Shows
    whether IMC conditions lead to more sever accidents.
    """
    if "weathercondition" not in df.columns or "highestinjurylevel" not in df.columns:
        return
    
    df_clean = df.copy()
    
    df_clean["weathercondition"] = df_clean["weathercondition"].astype(str).str.upper()

    severity_map = {
        "NONE": 0,
        "MINOR": 1, 
        "SERIOUS": 2,
        "FATAL": 3,
        "UNKNOWN": 4
    }

    df_clean["severity_score"] = df_clean["highestinjurylevel"].map(severity_map).fillna(4)

    grouped = df_clean.groupby("weathercondition")["severity_score"].mean()

    plt.figure()
    grouped.plot(kind = "bar")
    plt.title("Average Injury Severity: IMC vs VMC")
    plt.ylabel("Severity Score")

    if save:
        save_figure("imc_vs_vmc_risk")

    plt.show()
#=================================================================================

#=================================================================================
# plotting top dangerous aircraft models
def plot_top_aircraft_models(df, top_n = 10, save = False):
    """
    Identifies aircraft models most frequently involved in accidents.
    Assists in identifying higher-risk aircraft types.
    """
    if "make" not in df.columns or "model" not in df.columns:
        return
    
    df_clean = df.copy()

    df_clean["make"] = df_clean["make"].astype(str).str.upper().str.strip()

    df_clean["make"] = df_clean["make"].astype(str).str.upper().str.strip()
    df_clean["model"] = df_clean["model"].astype(str).str.upper().str.strip()

    df_clean["make_model"] = df_clean["make"] + " " + df_clean["model"]

    counts = df_clean["make_model"].value_counts().head(top_n)
    
    plt.figure()
    counts.plot(kind = "bar")
    plt.title(f"Top {top_n} Aircraft Models in Runway Excursions")
    plt.ylabel("Number of Events")

    if save:
        save_figure("top_aircraft_models")
    
    plt.show()
#=================================================================================

#=================================================================================
# injury severity vs aircraft damage
def plot_injury_vs_damage(df, normalize = True, save = False):
    """
    Visualizes the relationship between injury severity and aircraft 
    damage. Shows how aircraft damage relates to injury severity, and
    helps to identify potential patterns (high damage results in higher 
    injury severity, minor damage results in little to no injuries).
    """
    if "highestinjurylevel" not in df.columns or "aircraftdamage" not in df.columns:
        return
    
    df_clean = df.copy()

    df_clean["highestinjurylevel"] = (
        df_clean["highestinjurylevel"].astype(str).str.upper().str.strip()
    )
    df_clean["aircraftdamage"] = (
        df_clean["aircraftdamage"].astype(str).str.upper().str.strip()
    )

    # creating crosstab
    if normalize:
        table = pd.crosstab(
            df_clean["aircraftdamage"],
            df_clean["highestinjurylevel"],
            normalize = "index"
        )
        title = "Injury Severity Distribution by Aircraft Damage (Proportion)"
    else:
        table = pd.crosstab(
            df_clean["aircraftdamage"],
            df_clean["highestinjurylevel"]
        )
        title = "Injury Severity vs Aircraft Damage (Count)"

    plt.figure()
    table.plot(kind = "bar", stacked = True)
    plt.title(title)
    plt.xlabel("Aircraft Damage")
    plt.ylabel("Proportion" if normalize else "Count")
    plt.xticks(rotation = 45)

    if save: 
        save_figure("injury_vs_aircraft_damage")

    plt.show()
#=================================================================================

#=================================================================================
# risk scoring features
def plot_risk_score_distribution(df, save = False):
    """
    Creates a simple risk scoring system combining injury severity, weather conditions
    (IMC penalty), and wind speed contribution. Helps create features for
    ML modeling. 
    """
    required_cols = ["highestinjurylevel", "weathercondition", "sknt"]

    if not all(col in df.columns for col in required_cols):
        return

    df_clean = df.copy()

    severity_map = {
        "NONE": 0,
        "MINOR": 1, 
        "SERIOUS": 2,
        "FATAL": 3,
        "UNKNOWN": 4
    }

    df_clean["severity_score"] = df_clean["highestinjurylevel"].map(severity_map).fillna(4)

    # IMC penalty
    df_clean["imc_flag"] = df_clean["weathercondition"].astype(str).str.upper().apply(
        lambda x: 1 if "IMC" in x else 0
    )

    df_clean["wind_score"] = pd.to_numeric(df_clean["sknt"], errors = "coerce")

    # calculating risk score
    df_clean["risk_score"] = (
        df_clean["severity_score"] +
        df_clean["imc_flag"] + 
        (df_clean["wind_score"] / 10)
    )

    plt.figure()
    df_clean["risk_score"].dropna().hist(bins = 30)
    plt.title("Runway Excursion Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")

    if save:
        save_figure("risk_score_distribution")
    
    plt.show()
#=================================================================================

#=================================================================================
# events over time
def plot_events_over_time(df, save = False):
    """
    Plots number of runway excursions over time. Identifies trends and
    temporal patterns in accidents.
    """
    if "eventdate" not in df.columns:
        return
    
    df["eventdate"] = pd.to_datetime(df["eventdate"], errors = "coerce", utc = True)

    counts = df["eventdate"].dt.year.value_counts().sort_index()

    plt.figure()
    counts.plot()
    plt.title("Runway Excursions Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.xticks(rotation = 60)

    if save:
        save_figure("events_over_time")

    plt.show()
#=================================================================================

#=================================================================================
# correlation heatmap
def plot_correlation_heatmap(df, save = False):
    """
    Shows correlation among numeric METAR + engineered features. Helps
    identify redundancy and multicollinearity.
    """
    numeric_df = df.select_dtypes(include = [np.number])

    if numeric_df.empty:
        return
    
    plt.figure(figsize = (10, 8))
    corr = numeric_df.corr()

    sns.heatmap(corr, cmap = 'coolwarm', center = 0)

    plt.title("Correlation Heatmap (Numeric & Engineered Features)")

    if save:
        save_figure("correlation_heatmap")

    plt.show()
#=================================================================================

#=================================================================================
def crosswind_vs_tailwind(df, save = False):
    """
    Visualizes relationship between crosswind and tailwind components.
    """
    required = ["crosswind_component", "tailwind_component"]

    if not all(col in df.columns for col in required):
        return
    
    plt.figure()

    plt.scatter(df["crosswind_component"], df["tailwind_component"], alpha = 0.5)
    plt.title("Crosswind vs Tailwind Components")
    plt.xlabel("Crosswind Component (knots)")
    plt.ylabel("Tailwind Component (knots)")

    if save:
        save_figure("crosswind_vs_tailwind")

    plt.show()
#=================================================================================

#=================================================================================
def damage_rate_by_weather(df, save = False):
    """
    Shows proportion of severe damage (binary target) by weather condition.
    More informative than raw counts.
    """
    if "weathercondition" not in df.columns or "damage_binary" not in df.columns:
        return
    
    grouped = df.groupby("weathercondition")["damage_binary"].mean()

    plt.figure()
    grouped.plot(kind = "bar")

    plt.title("Damage Rate by Weather Condition")
    plt.ylabel("Proportion of Severe Damage")

    if save:
        save_figure("damage_rate_weather")

    plt.show()
#=================================================================================

#=================================================================================
def plot_log_transformed_distributions(df, save = False):
    """
    Shows log-transformed distributions to reveal fat tails
    in aviation environmental variables.
    """

    cols = ["sknt", "gust", "vsby", "p01i"]

    plt.figure(figsize=(10, 6))

    for i, col in enumerate(cols):
        if col in df.columns:
            plt.subplot(2, 2, i + 1)
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            plt.hist(np.log1p(data), bins=30)
            plt.title(f"log(1 + {col})")

    plt.tight_layout()

    if save:
        save_figure("log_transformed_distributions")

    plt.show()
#=================================================================================

#=================================================================================
def plot_class_imbalance(df, save = False):
    """
    Shows imbalance in runway excursion outcomes.
    """

    plt.figure()

    df["damage_binary"].value_counts(normalize=True).plot(kind="bar")

    plt.title("Class Imbalance: Runway Excursion Outcome")
    plt.xlabel("Class (0 = Non-severe, 1 = Severe)")
    plt.ylabel("Proportion")

    if save:
        save_figure("class_imbalance")

    plt.show()