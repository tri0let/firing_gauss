from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os

# ========= LOAD ==========
folder = "csv_data"


g_all = pd.read_csv(os.path.join(folder, "gaussian_cannon_all_stations_summary.csv"))
nm = pd.read_csv(os.path.join(folder, "no_magnets_summary.csv"))

# File structrure
for df in [ g_all, nm]:
    if "mean_speed" in df.columns:
        df["mean_speed"] = pd.to_numeric(df["mean_speed"], errors="coerce")
    if "sem_speed" in df.columns:
        df["sem_speed"] = pd.to_numeric(df["sem_speed"], errors="coerce")
    if "stations" in df.columns:
        df["stations"] = pd.to_numeric(df["stations"], errors="coerce")
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    if "magnets" in df.columns:
        df["magnets"] = pd.to_numeric(df["magnets"], errors="coerce")
    if "group" in df.columns:
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


g_all = split(g_all)
nm = split(nm)

full = pd.concat([g_all], ignore_index=True)

#======== CALCULATE DISTANCES ===========

