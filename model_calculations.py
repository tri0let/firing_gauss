import os
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# ========= LOAD ==========
folder = "csv_data"
g1 = pd.read_csv(os.path.join(folder, "gaussian_cannon_4-01-2026_summary.csv"))
g_all = pd.read_csv(os.path.join(folder, "gaussian_cannon_summary.csv"))
nm = pd.read_csv(os.path.join(folder, "no_magnets_summary.csv"))

# File structure
for df in [g1, g_all, nm]:
    df["mean_speed"] = pd.to_numeric(df["mean_speed"], errors="coerce")
    df["sem_speed"] = pd.to_numeric(df["sem_speed"], errors="coerce")
    df["stations"] = pd.to_numeric(df["stations"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["magnets"] = pd.to_numeric(df["magnets"], errors="coerce")
    df["group"] = df["group"].astype(str).str.strip()

# ========= SPLIT ENTRANCE VELOCITY / EXIT VELOCITY ==========
def split(df):
    vin = df[df["gate"] == "1+2"].copy()
    vout = df[df["gate"] == "3+4"].copy()

    merged = pd.merge(
        vin,
        vout,
        on=["group", "stations", "distance", "magnets"],
        suffixes=("_in", "_out")
    )
    return merged

g1 = split(g1)
g_all = split(g_all)
nm = split(nm)

full = pd.concat([g1, g_all], ignore_index=True)

# ========= REMOVE RUN 162-171 ==========
full = full[full["group"].str.strip() != "Run 162-171"].copy()
nm = nm[nm["group"].str.strip() != "Run 162-171"].copy()

print("\n===== CHECK REMOVAL =====\n")
print(full.loc[full["group"] == "Run 162-171", ["group", "stations", "distance", "magnets"]].to_string(index=False))

print(full)