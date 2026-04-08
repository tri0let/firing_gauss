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

# ========= BASELINE ==========
baseline_v = nm["mean_speed_out"].mean()

# ========= ANALYSIS ==========
full["dv"] = full["mean_speed_out"] - full["mean_speed_in"]

full["ratio"] = full["mean_speed_out"] / full["mean_speed_in"]

full["dv_sem"] = np.sqrt(
    full["sem_speed_in"]**2 + full["sem_speed_out"]**2
)

full["ratio_sem"] = full["ratio"] * np.sqrt(
    (full["sem_speed_in"] / full["mean_speed_in"])**2 +
    (full["sem_speed_out"] / full["mean_speed_out"])**2
)

full["gain_vs_nomagnet"] = full["mean_speed_out"] - baseline_v

full = full.sort_values(by=["stations", "distance", "magnets"]).reset_index(drop=True)

# MAXIMUM EXIT VELOCITY FOR EACH STATION 
max_per_station = (
    full.loc[full.groupby("stations")["mean_speed_out"].idxmax()]
    .sort_values(by="stations")
    .reset_index(drop=True)
)

print("\n===== MAXIMUM EXIT VELOCITY FOR EACH STATION COUNT =====\n")
print(
    max_per_station[
        ["stations", "distance", "magnets", "group", "mean_speed_out", "sem_speed_out"]
    ].to_string(index=False)
)

overall_max = full.loc[full["mean_speed_out"].idxmax()]

print("\n===== OVERALL MAXIMUM EXIT VELOCITY CONFIGURATION =====\n")
print(
    overall_max[
        ["stations", "distance", "magnets", "group", "mean_speed_out", "sem_speed_out"]
    ].to_string()
)

# TOP / WORST CONFIGURATIONS BASED ON Δv
top3 = full.nlargest(3, "dv")
worst3 = full.nsmallest(3, "dv")

print("\n===== TOP 3 CONFIGURATIONS (MAX Δv) =====\n")
print(top3[["group", "stations", "distance", "magnets", "dv", "dv_sem"]].to_string(index=False))

print("\n===== WORST 3 CONFIGURATIONS =====\n")
print(worst3[["group", "stations", "distance", "magnets", "dv", "dv_sem"]].to_string(index=False))

# SUMMARY TABLE OF ALL CONFIGURATIONS 
print("\n===== FINAL TABLE =====\n")
print(full[[
    "group", "stations", "distance", "magnets",
    "mean_speed_in", "mean_speed_out",
    "dv", "dv_sem",
    "ratio", "ratio_sem",
    "gain_vs_nomagnet"
]].to_string(index=False))

# Smooth spline function for plotting
def smooth_spline(x, y, lam=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return x, y
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    if len(x) != len(np.unique(x)):
        grouped = {}
        for xi, yi in zip(x, y):
            grouped.setdefault(xi, []).append(yi)
        x = np.array(sorted(grouped.keys()), dtype=float)
        y = np.array([np.mean(grouped[xi]) for xi in x], dtype=float)
    n = len(x)
    if n < 2:
        return x, y
    k = min(3, n - 1)
    spline = UnivariateSpline(x, y, k=k, s=2 if lam is None else lam)
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# PLOT 1: EXIT VELOCITY VS MAGNETS (FIXED STATIONS & DISTANCE) 
s = 1.0
d_station = full[full["stations"] == s].copy()

plt.figure(figsize=(8, 5))
for dist in sorted(d_station["distance"].dropna().unique()):
    d_line = d_station[d_station["distance"] == dist].copy()
    d_line = d_line.sort_values(by="magnets")

    if len(d_line) == 0:
        continue

    plt.errorbar(
        d_line["magnets"],
        d_line["mean_speed_out"],
        yerr=d_line["sem_speed_out"],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=6,
        label=f"distance = {dist} (cm)"
    )
    x_smooth, y_smooth = smooth_spline(d_line["magnets"], d_line["mean_speed_out"], lam=3)
    plt.plot(x_smooth, y_smooth, linewidth=1.8, alpha=0.8)


x_smooth_all, y_smooth_all = smooth_spline(d_station["magnets"], d_station["mean_speed_out"], lam=3)
plt.plot(x_smooth_all, y_smooth_all, linewidth=3, alpha=0.7, label="Spline - Fit")

plt.xlabel("Number of Magnets", fontsize=16)
plt.ylabel("Exit Velocity (m/s)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f" {int(s)} Station{'s' if s > 1 else ''}", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# PLOT 2: EXIT VELOCITY VS DISTANCE 
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
stations_to_plot = [2.0, 3.0]

for i, s in enumerate(stations_to_plot):
    ax = axes[i]
    d_station = full[full["stations"] == s].copy()

    for mag in sorted(d_station["magnets"].dropna().unique()):
        d_line = d_station[d_station["magnets"] == mag].copy()
        d_line = d_line.sort_values(by="distance")

        if len(d_line) == 0:
            continue

        ax.errorbar(
            d_line["distance"],
            d_line["mean_speed_out"],
            yerr=d_line["sem_speed_out"],
            fmt="o",
            capsize=4,
            linewidth=1.5,
            markersize=6,
            label=f"magnets = {int(mag)}"
        )
        x_smooth, y_smooth = smooth_spline(d_line["distance"], d_line["mean_speed_out"], lam=1)
        ax.plot(x_smooth, y_smooth, linewidth=1.8, alpha=0.8, label=f"Spline - {int(mag)} magnets")
    ax.set_xlabel("Distance (cm)", fontsize=16)
    ax.set_ylabel("Exit Velocity (m/s)", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f"({chr(97 + i)}) {int(s)} Station{'s' if s > 1 else ''}", fontsize=16)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

