"""
#Runs the full data pipeline
1. Load raw prompts
2. Clean text and compute features
3. Generate synthetic energy dataset
"""

import os
import numpy as np
import pandas as pd
from data_loader import load_dataset_raw
from data_pipeline import compute_feature
from energy_simulator import generate_energy_data


def main():
    print("=== Data Pipeline Started ===")

    #Ensures necessary directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/synthetic", exist_ok=True)

    # Load raw data
    print("Loading raw data...")
    df_raw = load_dataset_raw()
    print(f"Raw data loaded: {len(df_raw)} rows")

    #Clean and feature engineer
    print("Cleaning and computing features...")

    #Generatea synthetic numeric parameters
    n = len(df_raw)
    num_layers_list = np.random.randint(4, 48, size=n)
    training_hours_list = np.random.uniform(0.5, 20, size=n)
    flops_per_hour_list = np.random.uniform(1e9, 1e12, size=n)

    #Computes features for each row
    features_rows = []
    for i, row in df_raw.iterrows():
        features = compute_feature(
            row["prompt"],
            num_layers_list[i],
            training_hours_list[i],
            flops_per_hour_list[i]
        )
        features_rows.append(features)

    #Converts list of dicts to DataFrame
    df_features = pd.DataFrame(features_rows)

    # Saves processed features

    df_features.to_csv("data/processed/features_df.csv", index=False)
    print(f"Processed features saved: {df_features.shape[0]} rows")

    # generate synthetic energy dataset
    print("Generating synthetic energy data...")
    df_energy = generate_energy_data(df_features)
    df_energy.to_csv("data/synthetic/energy_dataset.csv", index=False)
    print(f"Synthetic energy dataset saved: {df_energy.shape[0]} rows")

    print("=== Data Pipeline Complete ===")


if __name__ == "__main__":
    main()
