import pandas as pd
import numpy as np

def generate_energy_data(df):
    #Generate synthetic energy labels for testing

    #Create synth energy consumption column for testing purposes

    df = df.copy()
    df["energy_kwh"] = (
        0.5
        + df["token_count"] * 0.003
        + df["avg_word_length"] * 0.10
        + np.random.normal(0,0.05, size=len(df)) #small noise
    )
    df.to_csv("data/synthetic/energy_dataset.csv", index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    generate_energy_data(df)
    print("Synthetic energy dataset saved to data/synthetic/energy_dataset.csv")