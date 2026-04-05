import pandas as pd
import re
import math
import os

# ========= FILE PATHS =======
folder = 'csv_data'
file_gaussian = os.path.join(
    folder,
    "04-02-2026_gaussian_cannon_all_stations[spped_time_between_stationsx191].csv"
)

file_nomagnet = os.path.join(
    folder,
    "3-31-2026_No_Magnets_Station[Speed_Time_Between_Gatesx7].csv"
)

# ========== Load Files ==========
df_gaussian = pd.read_csv(file_gaussian)
df_nomagnet = pd.read_csv(file_nomagnet)

# ========== EXTRACT FIRST VALID VALUE PER RUN ==========
def extract_first_values(df, gate):
    runs = sorted({
        int(m.group(1))
        for col in df.columns
        if (m := re.search(r"Run #(\d+)$", col))
    })

    records = []
    for run in runs:
        time_col = f"Time Between Gates, Ch {gate} (s) Run #{run}"
        speed_col = f"Speed Between Gates, Ch {gate} (m/s) Run #{run}"

        if time_col not in df.columns or speed_col not in df.columns:
            continue

        time_vals = df[time_col].dropna()
        speed_vals = df[speed_col].dropna()

        first_time = time_vals.iloc[0] if len(time_vals) > 0 else math.nan
        first_speed = speed_vals.iloc[0] if len(speed_vals) > 0 else math.nan

        records.append({
            "run": run,
            "gate": gate,
            "time": first_time,
            "speed": first_speed
        })

    out = pd.DataFrame(records)
    out = out.sort_values(by=["run", "gate"]).reset_index(drop=True)
    return out

# ========= REGROUP DATA INTO 10 TRIALS OF RUN (EG: Run 1-10, Run 11-20, ETC.) ======
def regroup_trials(df, block_size=10):
    df = df.copy()

    # Runs in ascending order
    unique_runs = sorted(df["run"].dropna().unique())

    # Split runs into chunks of 10
    run_to_group = {}
    run_to_group_start = {}
    run_to_group_end = {}

    for i in range(0, len(unique_runs), block_size):
        chunk = unique_runs[i:i+block_size]
        group_start = chunk[0]
        group_end = chunk[-1]
        group_label = f"Run {group_start}-{group_end}"

        for run in chunk:
            run_to_group[run] = group_label
            run_to_group_start[run] = group_start
            run_to_group_end[run] = group_end

    df["group_start"] = df["run"].map(run_to_group_start)
    df["group_end"] = df["run"].map(run_to_group_end)
    df["group"] = df["run"].map(run_to_group)

    df = df.sort_values(by=["run", "gate"]).reset_index(drop=True)
    return df

# ========= SUMMARY FUNCTION ======
def summarize(df):
    df = df.drop_duplicates(subset=["run", "gate"]).copy()

    summary = (
        df.groupby(["group_start", "group_end", "group", "gate"], sort=True)
        .agg(
            n_speed=("speed", "count"),
            mean_speed=("speed", "mean"),
            std_speed=("speed", "std"),
            n_time=("time", "count"),
            mean_time=("time", "mean"),
            std_time=("time", "std"),
        )
        .reset_index()
    )

    summary["sem_speed"] = summary["std_speed"] / summary["n_speed"]**0.5
    summary["sem_time"] = summary["std_time"] / summary["n_time"]**0.5

    summary = summary.sort_values(by=["group_start", "gate"]).reset_index(drop=True)
    return summary

# ========= Process Gaussian File =========
g12 = extract_first_values(df_gaussian, "1+2")
g34 = extract_first_values(df_gaussian, "3+4")

gaussian_file = pd.concat([g12, g34], ignore_index=True)
gaussian_file = gaussian_file.sort_values(by=["run", "gate"]).reset_index(drop=True)
gaussian_file = regroup_trials(gaussian_file, block_size=10)
gaussian_summary = summarize(gaussian_file)

gaussian_file = gaussian_file[["run", "gate", "time", "speed", "group_start", "group_end", "group"]]

gaussian_summary = gaussian_summary[
    [
        "group",
        "gate",
        "n_speed",
        "mean_speed",
        "std_speed",
        "sem_speed",
        "n_time",
        "mean_time",
        "std_time",
        "sem_time"
    ]
]

# ========= Process No-Magnet File =========
g12 = extract_first_values(df_nomagnet, "1+2")
g34 = extract_first_values(df_nomagnet, "3+4")

nomagnet_file = pd.concat([g12, g34], ignore_index=True)
nomagnet_file = nomagnet_file.sort_values(by=["run", "gate"]).reset_index(drop=True)

# keep all 7 runs together
nomagnet_file["group_start"] = nomagnet_file["run"].min()
nomagnet_file["group_end"] = nomagnet_file["run"].max()
nomagnet_file["group"] = "All Runs"

nomagnet_summary = summarize(nomagnet_file)

nomagnet_file = nomagnet_file[["run", "gate", "time", "speed", "group_start", "group_end", "group"]]

nomagnet_summary = nomagnet_summary[
    [
        "group",
        "gate",
        "n_speed",
        "mean_speed",
        "std_speed",
        "sem_speed",
        "n_time",
        "mean_time",
        "std_time",
        "sem_time"
    ]
].sort_values(by=["gate"]).reset_index(drop=True)

# ========== Output Files =========
gaussian_file.to_csv(os.path.join(folder, "gaussian_cannon_all_stations.csv"), index=False)
gaussian_summary.to_csv(os.path.join(folder, "gaussian_cannon_summary.csv"), index=False)
nomagnet_file.to_csv(os.path.join(folder, "no_magnets_station.csv"), index=False)
nomagnet_summary.to_csv(os.path.join(folder, "no_magnets_summary.csv"), index=False)

print("=============== Gaussian Groups ===============")
print(
    gaussian_file[["group_start", "group_end", "group"]]
    .drop_duplicates()
    .sort_values(by="group_start")
    .to_string(index=False)
)

print("============= Gaussian Trials ==============")
print(gaussian_file.head(30).to_string(index=False))

print("============= Gaussian Summary ==============")
print(gaussian_summary.to_string(index=False))

print("============== No-Magnets Summary ================")
print(nomagnet_summary.to_string(index=False))

# ========= OPTIONAL CHECK ======
example_group = gaussian_file["group"].drop_duplicates().iloc[0]
print(f"CHECK GROUP: {example_group}")
check = gaussian_file[gaussian_file["group"] == example_group]
print(check.to_string(index=False))

print("\nCounts per gate in that group:")
print(check.groupby("gate").size())