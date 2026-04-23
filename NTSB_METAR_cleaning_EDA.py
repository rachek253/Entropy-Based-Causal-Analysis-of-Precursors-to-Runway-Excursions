"""
File Name: NTSB_METAR_cleaning_EDA.py

Purpose: This script runs the full exploratory data analysis (EDA) 
pipeline for the NTSB & METAR runway excursion dataset. This script
integrates data loading and cleaning utilities, descriptive statistical
analysis, visualization genration, statistical hypothesis testing 
(chi-square, t-test), and ranks feature importance using a baseline ML
model.
"""
import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv

# installing pipeline modules
from pipeline.cleaning import (
    drop_unnecessary_columns,
    standardize_types,
    fill_missing_values,
    add_wind_components
)

from pipeline.eda import (
    dataset_overview,
    missing_value_summary,
    numeric_summary,
    categorical_summary,
    aircraft_insights,
    weather_source_summary,
    outcome_severity,
    geographical_summary,
    operation_summary,
    weather_condition_summary,
    time_alignment_quality,
    run_chi_square, 
    run_ttest
)

from pipeline.visualization import (
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
    plot_correlation_heatmap,
    crosswind_vs_tailwind, 
    damage_rate_by_weather,
    plot_log_transformed_distributions,
    plot_class_imbalance
)

from pipeline.feature_engineering import (
    build_encoded_dataset,
    mutual_information_ranking,
    sklearn_mi_ranking,
    cmi_feature_ranking, 
    select_topk_precursors,
    compare_mi_cmi
)
#=================================================================================
load_dotenv()
base_path = os.getenv("PROJECT_DATA_DIRECTORY")
if not base_path:
    raise ValueError("PROJECT_DATA_DIRECTORY is not set. Check your .env file.")

input_path = os.path.join(base_path, "FINAL_NTSB_METAR_DATASET.csv")
cleaned_path = os.path.join(base_path, "NTSB_METAR_cleaned.csv")
#=================================================================================
# main pipeline for cleaning, EDA, visualization, and feature selection
def main():
    """
    Runs full EDA pipeline, which includes:
        - Cleaning data
        - Descriptive statistics to understand dataset
        - Visualizations
        - Statistical tests
        - Feature selection (information-theoretic)

    To use main function:
        - Remove quotations around section to be run (EDA, Chi-square
        & T tests, data visualization)
    """
    # loading full dataset
    df = pd.read_csv(input_path)

    #=================================================================
    # data cleaning
    df = drop_unnecessary_columns(df)
    df = standardize_types(df)
    df = fill_missing_values(df)
    df = add_wind_components(df)

    # saving cleaned data to its own .csv to not lose merged data
    df.to_csv(cleaned_path, index = False)

    df["weathercondition"] = df["weathercondition"].str.upper()

    
    # creating target variable for feature selection
    df["damage_binary"] = df["aircraftdamage"].apply(
    lambda x: 1 if x in ["Substantial", "Destroyed"] else 0)

    target_col = "damage_binary"
    #=================================================================
    # EDA (descriptive statistics)
    """
    dataset_overview(df)
    missing_value_summary(df)
    numeric_summary(df)
    categorical_summary(df)
    aircraft_insights(df)
    weather_source_summary(df)
    outcome_severity(df)
    geographical_summary(df)
    operation_summary(df)
    weather_condition_summary(df)
    time_alignment_quality(df)
    """
    #=================================================================
    # chi-square test
    """
    chi_square_results = []
    categorical_features = ["weathercondition", "purposeofflight",
                            "highestinjurylevel", "far", "operator",
                            "make", "model"]
    for col in categorical_features:
        result = run_chi_square(df, col, "damage_binary")
        if result:
            chi2, p, dof = result
            chi_square_results.append((col, chi2, p))

    # t test
    ttest_results = []
    # creating weather_numeric cols
    weather_numeric_cols = ["tmpf", "dwpf", "relh", "drct", "sknt",
                            "gust", "vsby", "alti", "p01i", 
                            "crosswind_component", "tailwind_component"]
    for col in weather_numeric_cols:
        result = run_ttest(df, col, "damage_binary", 0, 1)
        if result is not None:
            t_stat, p = result
            ttest_results.append((col, t_stat, p))

    chi_df = pd.DataFrame(chi_square_results, columns=["feature", "chi2", "p_value"])
    ttest_df = pd.DataFrame(ttest_results, columns=["feature", "t_stat", "p_value"])

    print("\nTop Chi-Square Features:")
    print(chi_df.sort_values("p_value").head(10))

    print("\nTop T-Test Features:")
    print(ttest_df.sort_values("p_value").head(10))
    """
    #=================================================================
    # data visualization
    """ 
    plot_injury_severity(df, save = True)
    plot_aircraft_damage(df, save = True )
    plot_weather_conditions(df, save = True)
    plot_wind_speed(df, save = True)
    plot_visibility_vs_severity(df)
    plot_imc_vs_vmc_risk(df)
    plot_temperature_distribution(df, save = True)
    plot_risk_score_distribution(df, save = True)
    plot_top_aircraft_models(df)
    plot_injury_vs_damage(df, save = True)
    plot_correlation_heatmap(df, save = True)
    crosswind_vs_tailwind(df, save = True)
    damage_rate_by_weather(df, save = True)
    plot_log_transformed_distributions(df, save = True)
    plot_class_imbalance(df, save = True)
    """
    #=================================================================
    # feature selection
    """
    print("\n=========== Feature Selection for Information-Theoretic Analysis ==========")

    # build encoded dataset
    df_encoded = build_encoded_dataset(df, target_col)

    # custom MI
    mi_custom = mutual_information_ranking(df_encoded, target_col)
    topk_cust = select_topk_precursors(mi_custom, top_k = 10)
    print("\nTop MI Features (Custom):")
    print(topk_cust)

    # sklearn MI baseline comparison
    mi_sklearn = sklearn_mi_ranking(df_encoded, target_col)
    topk_sklearn = select_topk_precursors(mi_sklearn, top_k = 10)
    print("\nTop MI Features (Sklearn):")
    print(topk_sklearn)

    # MI agreement analysis
    overlap = list(set(topk_cust).intersection(set(topk_sklearn)))
    print("\n Overlapping Top Features (Robust Predictors):")
    print(overlap)

    # CMI feature selection (greedy selection)
    cmi_features, cmi_scores = cmi_feature_ranking(df_encoded, target_col)

    # enforcing <=10 features to match report constraint
    cmi_features = cmi_features[:10]
    cmi_scores = cmi_scores[:10]

    print("\nTop Features (Conditional Mutual Information - Greedy Selection Order):")
    for f, s in zip(cmi_features, cmi_scores):
        print(f"{f}: {s:.4f}")

    # MI vs CMI comparison table
    comparison_MI_CMI = compare_mi_cmi(df_encoded, target_col)
    print("\n======== MI vs CMI Feature Selection Comparison =======")
    print(comparison_MI_CMI)
    """
#=================================================================================   
if __name__ == "__main__":
    main()