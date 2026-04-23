"""
File Name: causal_analysis_main.py

Purpose: This script performs information-theoretic causal discovery 
using Conditional Mutual Informaiton (CMI), Causation Entropy (forward
selection), permutation-based significance testing, and bootstrap
stability analysis. This script assumes that the NTSB/METAR dataset has
been preprocessed and important precursors to runway excursions have 
been selected using Mutual Information (MI)/CMI. All core logic is 
implemented in the causal_analysis package, and this file serves as only
the entry point. 
"""
import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from causal_analysis.utils import create_target
from causal_analysis.core_CSE import run_analysis
from causal_analysis.causal_graph import build_causal_graph, plot_graph
from causal_analysis.latex_tables import create_latex_tables, save_latex_tables
#=================================================================================
# configuration for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
#=================================================================================

#=================================================================================
# load merged clean dataset
load_dotenv()
base_path = os.getenv("PROJECT_DATA_DIRECTORY")
if not base_path:
    raise ValueError("PROJECT_DATA_DIRECTORY is not set. Check your .env file.")

input_path = os.path.join(base_path, "NTSB_METAR_cleaned.csv")
#=================================================================================

#=================================================================================
def main():
    """
    Main execution function for causal analysis pipeline.

    Steps include:
    1. Loading NTSB/METAR dataset
    2. Create binary target variable
    3. Run causation entropy feature selection
    4. Perform permutation testing
    5. Compute bootstrap stability
    6. Build causal graph (edge list)
    7. Export LaTex tables
    8. Visualize causal graph (use save = True to save figure)
    """
    # loading NTSB/METAR dataset
    df = pd.read_csv(input_path)

    # creating binary_damage as target variable
    df = create_target(df)

    # running analysis
    results = run_analysis(df)

    print("\n======== Results ========")
    print("\nSelected Features (Causation Entropy Order):")

    for  f in results["selected_features"]:
        print(f"{f}: {results['scores'][f]:.4f}")

    print("\nPermutation Test p-values:")
    for f, p in results["p_values"].items():
        print(f"{f}: {p:.4f}")

    print("\nBootstrap Stability (selection frequency):")
    for f, s in results["stability"].items():
        print(f"{f}: {s:.2f}")

    print("\n======== Causal Graph (Edges) =======")
    edges = build_causal_graph(results)
    for e in edges:
        print(f"{e['source']} -> {e['target']} (weight = {e.get('weight', 0):.4f})")

    plot_graph(edges, save = True)

    # exporting tables to LaTex
    main_table, stability_table, graph_table = create_latex_tables(
        results,
        edges
    )

    save_latex_tables(
        main_table,
        stability_table,
        graph_table,
        output_dir = "outputs"   
        )
#=================================================================================

#=================================================================================
if __name__ == "__main__":
    main()