"""
File Name: __init__.py

Purpose: Initializes the pipeline folder to be read as a Python package.
This allows modules within the pipeline (cleaning, EDA, and visualization)
to be imported easily across scripts. 

Modules Included:
    - cleaning: Utility functions for data cleaning and preprocessing
    - eda: Exploratory data analysis (EDA) functions
    - visualization: Data visualization tools for analysis
    - feature_engineering: Engineers features for feature selection

Usage Example:
    from pipline import cleaning, eda, visualization

    df = cleaning.drop_unnecessary_columns(df)
    eda.dataset_overview(df)
    visualization.plot

    # importing key modules so they can be accessed directly
from . import cleaning
from . import eda
from . import visualization
from . import feature_engineering

# common functions at a package level
from .cleaning import (
    drop_unnecessary_columns,
    standardize_types,
    fill_missing_values,
    add_wind_components
)

from .eda import (
    dataset_overview,
    missing_value_summary,
    numeric_summary,
    categorical_summary, 
    aircraft_insights,
    weather_source_summary,
    time_alignment_quality,
    outcome_severity,
    geographical_summary,
    operation_summary,
    weather_condition_summary
)

from .visualization import (
    save_figure,
    plot_missing_values,
    plot_injury_severity,
    plot_aircraft_damage,
    plot_weather_conditions,
    plot_temperature_distribution,
    plot_wind_speed,
    plot_visibility_vs_severity,
    plot_imc_vs_vmc_risk,
    plot_top_aircraft_models,
    plot_injury_vs_damage,
    plot_risk_score_distribution,
    plot_events_over_time
)

from .feature_engineering import(
    build_encoded_dataset,
    mutual_information_ranking, 
    sklearn_mi_ranking,
    select_top8_precursors
)
"""
