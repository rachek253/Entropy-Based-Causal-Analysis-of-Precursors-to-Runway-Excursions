# Runway Excursion Causal Analysis 

This project performs information-theoretic causal discovery on aviation runway excursion events using
NTSB accident data and METAR weather observations. The goal of this project is to identify statistically
meaningful precursor variables that contribute to runway excursion risk using mutual information (MI), 
conditional mutual information (CMI), and causation entropy (CSE).

-----------

## Project Overview

Runway excursions are among the most common aviation safety events, occurring when an aircraft departs 
the runway surface during landing or takeoff. This project investigates how operational, environmental, 
and aircraft-related variables interact to contribute to these events using a data-driven, 
information-theoretic framework.

This analysis does **not assume causal structure**, but instead infers candidate dependency
relationships from observational data.

-----------

## Research Objectives

The primary objectives of this project are:
  - Identify variables that contain information about runway excursion outcomes
  - Estimate conditional dependencies using MI, CMI, and CSE
  - Construct a directed dependency graph of candidate precursor relationships
  - Evaluate robustness using permutation testing and bootstrap resampling
  - Assess stability of inferred relationships under sampling and estimator variation

-----------

## Repository Structure

data/
  - raw/
    * NTSB Accident Data/
    * Weather Data/
  - processed/
    * Airport Stations/
    * METAR/
    * Merged Datasets/
pipeline/
  - __init__.py
  - cleaning.py
  - feature_engineering.py
  - visualization.py
information_theory/
  - __init__.py
  - Entropy.py
  - Joint_Entropy.py
  - Mutual_Information.py
  - Conditional_Mutual_Information.py
  - Casuation_Entropy.py
  - cmi_FeatureSelection.py
causal_analysis/
  - __init__.py
  - utils.py
  - core_CSE.py
  - causal_graph.py
  - latex_tables.py
analysis_scripts/
  - NTSB_METAR_cleaning_EDA.py
  - causal_analysis_main.py
METAR Scripts/
  - METAR.py
  - METAR_fallback_stations.py
  - METAR_full.py
  - METAR_event_aligned_merge.py
figures/
  - figures_png
  - figures_svg
Project Documentation/
  - Entropy_Based_Causal_Analysis_of_Precursors_to_Runway_Excursions.pdf

-----------

## Methodology

The pipeline includes:
  - Data preprocessing and NTSB-METAR event matching
  - Feature engineering (cross-wind, tailwind, log-transformed variables)
  - Mutual Information (MI) feature ranking
  - Conditional mutual information (CMI) greedy selection
  - Causation entropy (CSE) causal discovery
  - Permutation testing for statistical significance
  - Bootstrap stability analysis for robustness
  - Directed causal graph construction

-----------

## Core Methods

### Mutual Information (MI)
Measures marginal dependency between variables using empirical probability distributions.

### Conditional Mutual Information (CMI)
Quantifies dependency between variables while conditioning on previously selected features.

### Causation Entropy (CSE)
Extends CMI in a greedy forward-selection framework to construct directed dependency structures.

### Permutation Testing
Statistical significance is evaluated by permuting the target variable to generate a null 
distribution of MI/CMI/CSE values.

### Bootstrap Sampling
Feature stability is evaluated by repeatedly resampling the dataset and re-running the causal 
discovery pipeline.

-----------

## Main Scripts

### `NTSB_METAR_cleaning_EDA.py`
  - Performs data cleaning and preprocessing
  - Handles missing values and categorical encoding
  - Generates exploratory plots and summary statistics
  - Ranked precursor tables for MI and CMI

### `causal_analysis_main.py`
  - Runs full causal discovery pipeline
  - Performs MI, CMI, and CSE estimation
  - Outputs feature rankings and causal graph

-----------

## Key Outputs  
  - Ranked precursor feature tables (MI/CMI/CSE scores)
  - Directed causal dependency graph
  - Bootstrap stability scores
  - Permutation test p-values
  - LaTex-formatted tables for paper integration
  - Saved figures in `/figures`

-----------

## Reproducibility
All experiments are fully reproducible using a fixed random seed:

```python
RANDOM_SEED = 0
```

-----------

## Environment Setup

### Requirements
Install dependencies: 
pip install -r requirements.txt

Required libraries include:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - networkx
  - python-dotenv
  - os

### Environment 
Create a .env file using:
  - `.env.example`

Create a .env file and copy the contents of the example .env. replace /path/to/data with your 
project directory. Ensure that you keep the .env file in the same path directory as the main script
files to ensure the project can run.

-----------

## Computational Cost

The comupational cost of the pipeline is moderate due to METAR API integration, bootstrap resampling,
and permutation testing. 

The most expensive components are:
  - Repeated estimation of MI/CMI/CSE across greedy selection iterations
  - Bootstrap resampling of full causal pipeline
  - Permutation testing for statistical dependence

Despite this, runtime for the project remains manageable due to the limited dataset size (428 events)
and vectorized NumPy-based implementation. Memory usage is controlled by on-demand computation
of joint distributions rather than precomputed high-dimensional tensors.

-----------

## Limitations

  - Limited sample size (428 events)
  - Sensitivity to discretization choice
  - Potential confounding from unobserved operational variables
  - Temporal mismatch between METAR observations and event timestamps
  - Observational nature prevents definitive causal claims

-----------

## Data Sources

  - NTSB Aviation Accident Database
  - Iowa State Mesonet ASOS/METAR Archive
  - National Weather ASOS stations list

-----------

## Citation 
If you use this work, please cite appropriately or reference the repository.

-----------

## Author Notes

This project is intended as an exploratory causal inference framework for aviation safety analysis. 
Results should be interpreted as statistically supported dependencies rather than confirmed
physical causation of runway excursion events.








