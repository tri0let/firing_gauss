import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= LOAD ==========
folder = "csv_data"
g1 = pd.read_csv(os.path.join(folder, "gaussian_cannon_4-01-2026_summary.csv"))
g_all = pd.read_csv(os.path.join(folder, "gaussian_cannon_summary.csv"))
nm = pd.read_csv(os.path.join(folder, "no_magnets_summary.csv"))

# File structrureL 
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

# MAXIMUM EXIT VELOCITY FOR EACH STATION COUNT
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


# PLOT 1: EXIT VELOCITY VS MAGNETS (FIXED STATIONS & DISTANCE)
for s in sorted(full["stations"].dropna().unique()):
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
            fmt="o-",
            capsize=4,
            linewidth=1.5,
            markersize=6,
            label=f"distance = {dist}"
        )

    # mark station maximum
    idx = d_station["mean_speed_out"].idxmax()
    row = d_station.loc[idx]
    plt.annotate(
        f"max = {row['mean_speed_out']:.3f}",
        (row["magnets"], row["mean_speed_out"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )

    plt.xlabel("Number of Magnets")
    plt.ylabel("Exit Velocity (m/s)")
    plt.title(f"Exit Velocity vs Magnets ({int(s)} Station{'s' if s > 1 else ''})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# PLOT 2: EXIT VELOCITY VS DISTANCE
for s in sorted(full["stations"].dropna().unique()):
    d_station = full[full["stations"] == s].copy()

    plt.figure(figsize=(8, 5))

    for mag in sorted(d_station["magnets"].dropna().unique()):
        d_line = d_station[d_station["magnets"] == mag].copy()
        d_line = d_line.sort_values(by="distance")

        if len(d_line) == 0:
            continue

        plt.errorbar(
            d_line["distance"],
            d_line["mean_speed_out"],
            yerr=d_line["sem_speed_out"],
            fmt="o-",
            capsize=4,
            linewidth=1.5,
            markersize=6,
            label=f"magnets = {int(mag)}"
        )
    # mark station maximum
    idx = d_station["mean_speed_out"].idxmax()
    row = d_station.loc[idx]
    plt.annotate(
        f"max = {row['mean_speed_out']:.3f}",
        (row["distance"], row["mean_speed_out"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )
    plt.xlabel("Distance")
    plt.ylabel("Exit Velocity (m/s)")
    plt.title(f"Exit Velocity vs Distance ({int(s)} Station{'s' if s > 1 else ''})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# PLOT 3: VELOCITY GAIN VS MAGNETS (FIXED STATIONS AND DISTANCE)
for s in sorted(full["stations"].dropna().unique()):
    d_station = full[full["stations"] == s].copy()

    plt.figure(figsize=(8, 5))

    for dist in sorted(d_station["distance"].dropna().unique()):
        d_line = d_station[d_station["distance"] == dist].copy()
        d_line = d_line.sort_values(by="magnets")

        if len(d_line) == 0:
            continue

        plt.errorbar(
            d_line["magnets"],
            d_line["dv"],
            yerr=d_line["dv_sem"],
            fmt="o-",
            capsize=4,
            linewidth=1.5,
            markersize=6,
            label=f"distance = {dist}"
        )

    plt.xlabel("Number of Magnets")
    plt.ylabel("Velocity Gain Δv (m/s)")
    plt.title(f"Velocity Gain vs Magnets ({int(s)} Station{'s' if s > 1 else ''})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# PLOT 4: RATIO VS DISTANCE (STATIONS FIXED )
for s in sorted(full["stations"].dropna().unique()):
    d_station = full[full["stations"] == s].copy()
    plt.figure(figsize=(8, 5))
    for mag in sorted(d_station["magnets"].dropna().unique()):
        d_line = d_station[d_station["magnets"] == mag].copy()
        d_line = d_line.sort_values(by="distance")

        if len(d_line) == 0:
            continue
        plt.errorbar(
            d_line["distance"],
            d_line["ratio"],
            yerr=d_line["ratio_sem"],
            fmt="o-",
            capsize=4,
            linewidth=1.5,
            markersize=6,
            label=f"magnets = {int(mag)}"
        )
    plt.xlabel("Distance")
    plt.ylabel("Velocity Ratio (v_out / v_in)")
    plt.title(f"Velocity Ratio vs Distance ({int(s)} Station{'s' if s > 1 else ''})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# PLOT 5: VELOCITY GAIN VS NO MAGNETS
plt.figure(figsize=(8, 5))
for s in sorted(full["stations"].dropna().unique()):
    d_station = full[full["stations"] == s].copy()
    plt.scatter(
        d_station["magnets"],
        d_station["gain_vs_nomagnet"],
        label=f"{int(s)} station{'s' if s > 1 else ''}"
    )

plt.xlabel("Number of Magnets")
plt.ylabel("Velocity Gain vs No-Magnet Baseline (m/s)")
plt.title("In comparison to the No-Magnet Baseline")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
