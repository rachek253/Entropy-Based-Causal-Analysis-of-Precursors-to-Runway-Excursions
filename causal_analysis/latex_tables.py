"""
File Name: latex_tables.py

Purpose: This module converts causal analysis results into report-ready
LaTex tables. LaTex tables are created for the main causal analysis 
results, the bootstrap stability, and the causal graph created for the 
analysis and are stored on the designated directory path.
"""
import os
import pandas as pd
#=================================================================================
def create_latex_tables(results, edges):
    """
    Converts results and causal graph into pandas DataFrames.
    """
    selected = results["selected_features"]
    scores = results["scores"]
    p_vals = results["p_values"]
    stability = results["stability"]

    main_table = pd.DataFrame({
        "Feature": selected,
        "CSE Score": [scores[f] for f in selected],
        "p_value": [p_vals.get(f, None) for f in selected],
        "Stability": [stability.get(f, None) for f in selected]
    })

    main_table = main_table.sort_values(by = "CSE Score", ascending = False)

    stability_table = pd.DataFrame(
        list(stability.items()),
        columns = ["Feature", "Selection Frequnecy"]
    )

    graph_table = pd.DataFrame(edges)

    return main_table, stability_table, graph_table
#=================================================================================

#=================================================================================
def save_latex_tables(main_table, stability_table, graph_table, output_dir = "outputs"):
    """
    Saves DataFrames as LaTex ready tables.
    """
    os.makedirs(output_dir, exist_ok = True)

    # saving LaTex table for results
    main_latex = main_table.to_latex(
        float_format = "%.4f",
        longtable = True,
        escape = False,
        caption = "Causation Entropy Feature Selection Results",
        label = "tab:cse_results",
        index = False
    )

    with open(os.path.join(output_dir, "main_results.tex"), "w") as f:
        f.write(main_latex)

    # stablility table
    stability_latex = stability_table.to_latex(
        float_format = "%.3f",
        longtable = True, 
        escape = False,
        caption = "Bootstrap Stability Candidate Featues",
        label = "tab:stability",
        index = False
    )

    with open(os.path.join(output_dir, "stability.tex"), "w") as f:
        f.write(stability_latex)

    graph_table = graph_table[graph_table["target"] == "damage_binary"]

    # causal graph table
    graph_latex = graph_table.to_latex(
        float_format = "%.4f",
        longtable = True,
        escape = False,
        caption = "Causal Graph Edge List Derived From Causation Entropy",
        label = "tab:causal_graph",
        index = False
    )

    with open(os.path.join(output_dir, "causal_graph.tex"), "w") as f:
        f.write(graph_latex)
#=================================================================================
