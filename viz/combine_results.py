import pandas as pd
import os

conv = 'figures/supplementary/convergence_aggregated.csv'
lat  = 'figures/supplementary/latency_vs_load_data.csv'
gap  = 'figures/supplementary/optimality_gap_data.csv'

dfs = []
if os.path.exists(conv):
    dfs.append(pd.read_csv(conv))
if os.path.exists(lat):
    dfs.append(pd.read_csv(lat))
if os.path.exists(gap):
    dfs.append(pd.read_csv(gap))

if dfs:
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged.to_csv('figures/supplementary/all_results_combined.csv', index=False)
    print("Combined CSV written to figures/supplementary/all_results_combined.csv")
else:
    print("No result CSVs found.")
