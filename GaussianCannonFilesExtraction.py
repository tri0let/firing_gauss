import pandas as pd
import re
import math
import os

# ========= FILE PATHS ==========
folder = "csv_data"

# Gaussian cannon first station file
file_gaussian = os.path.join(
    folder,
    "4-01-2026_gaussian_cannon_1st_station[speed_time_between_gatesx40].csv"
)

# Gaussian cannon 2nd and 3rd stations file
file_gaussian_1 = os.path.join(
    folder,
    "04-02-2026_gaussian_cannon_all_stations[speed_time_between_stationsx191].csv"
)

# No-magnets file
file_nomagnet = os.path.join(
    folder,
    "3-31-2026_No_Magnets_Station[Speed_Time_Between_Gatesx7].csv"
)

# ========= LOAD FILES ==========
df_gaussian = pd.read_csv(file_gaussian)
df_gaussian_1 = pd.read_csv(file_gaussian_1)
df_nomagnet = pd.read_csv(file_nomagnet)


# ========= EXTRACT FIRST VALID VALUE PER RUN ==========
def extract_first_values(df, gate):
    """
    For one gate pair (for example '1+2' or '3+4'),
    extract the first non-NaN time and speed value from each run.
    """
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


# ========= FIXED GROUPING FOR GAUSSIAN DATA ==========
def assign_gaussian_group(run):
    """
    Assign runs to the intended blocks.

    Before 141:
        1-10, 11-20, 21-30, ...

    Special block:
        141-151

    After that:
        152-161, 162-171, 172-181, ...
    """
    if pd.isna(run):
        return pd.Series([math.nan, math.nan, None])

    run = int(run)

    if run <= 140:
        start = ((run - 1) // 10) * 10 + 1
        end = start + 9
        return pd.Series([start, end, f"Run {start}-{end}"])

    if 141 <= run <= 151:
        return pd.Series([141, 151, "Run 141-151"])

    start = 152 + ((run - 152) // 10) * 10
    end = start + 9
    return pd.Series([start, end, f"Run {start}-{end}"])


def regroup_gaussian_trials(df):
    """
    Apply intended Gaussian run grouping based on actual run numbers,
    not by chunks of existing rows.
    """
    df = df.copy()
    df[["group_start", "group_end", "group"]] = df["run"].apply(assign_gaussian_group)
    df["group_start"] = pd.to_numeric(df["group_start"], errors="coerce")
    df["group_end"] = pd.to_numeric(df["group_end"], errors="coerce")
    df = df.sort_values(by=["run", "gate"]).reset_index(drop=True)
    return df


# ========= GROUPING FOR NO-MAGNET DATA ==========
def regroup_nomagnet_trials(df):
    """
    Keep all no-magnet runs together in one group.
    """
    df = df.copy()
    df["group_start"] = df["run"].min()
    df["group_end"] = df["run"].max()
    df["group"] = "All Runs"
    df = df.sort_values(by=["run", "gate"]).reset_index(drop=True)
    return df


# ========= SUMMARY FUNCTION ==========
def summarize(df):
    """
    Summarize by group and gate.
    """
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

    summary["sem_speed"] = summary["std_speed"] / summary["n_speed"] ** 0.5
    summary["sem_time"] = summary["std_time"] / summary["n_time"] ** 0.5

    summary = summary.sort_values(by=["group_start", "gate"]).reset_index(drop=True)
    return summary


# ========= PROCESS GAUSSIAN FILE 1 ==========
g1_12 = extract_first_values(df_gaussian, "1+2")
g1_34 = extract_first_values(df_gaussian, "3+4")

gaussian_part1 = pd.concat([g1_12, g1_34], ignore_index=True)
gaussian_part1 = gaussian_part1.sort_values(by=["run", "gate"]).reset_index(drop=True)

# ========= PROCESS GAUSSIAN FILE 2 ==========
g2_12 = extract_first_values(df_gaussian_1, "1+2")
g2_34 = extract_first_values(df_gaussian_1, "3+4")

gaussian_part2 = pd.concat([g2_12, g2_34], ignore_index=True)
gaussian_part2 = gaussian_part2.sort_values(by=["run", "gate"]).reset_index(drop=True)

# ========= COMBINE BOTH GAUSSIAN FILES ==========
gaussian_file = pd.concat([gaussian_part1, gaussian_part2], ignore_index=True)

# Remove accidental duplicates if the same run/gate appears in both files
gaussian_file = gaussian_file.drop_duplicates(subset=["run", "gate"], keep="first")

gaussian_file = gaussian_file.sort_values(by=["run", "gate"]).reset_index(drop=True)

# Apply corrected grouping
gaussian_file = regroup_gaussian_trials(gaussian_file)
gaussian_summary = summarize(gaussian_file)

gaussian_file = gaussian_file[
    ["run", "gate", "time", "speed", "group_start", "group_end", "group"]
]

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
# ========= PROCESS + SAVE FIRST GAUSSIAN FILE (4-01-2026) ==========
gaussian_part1 = regroup_gaussian_trials(gaussian_part1)
gaussian_part1_summary = summarize(gaussian_part1)

gaussian_part1 = gaussian_part1[
    ["run", "gate", "time", "speed", "group_start", "group_end", "group"]
]

gaussian_part1_summary = gaussian_part1_summary[
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


# ========= PROCESS NO-MAGNET FILE ==========
n12 = extract_first_values(df_nomagnet, "1+2")
n34 = extract_first_values(df_nomagnet, "3+4")

nomagnet_file = pd.concat([n12, n34], ignore_index=True)
nomagnet_file = nomagnet_file.sort_values(by=["run", "gate"]).reset_index(drop=True)

nomagnet_file = regroup_nomagnet_trials(nomagnet_file)
nomagnet_summary = summarize(nomagnet_file)

nomagnet_file = nomagnet_file[
    ["run", "gate", "time", "speed", "group_start", "group_end", "group"]
]

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


# ========= SAVE OUTPUT FILES ==========
gaussian_file.to_csv(os.path.join(folder, "gaussian_cannon_all_stations.csv"), index=False)
gaussian_summary.to_csv(os.path.join(folder, "gaussian_cannon_summary.csv"), index=False)
gaussian_part1.to_csv(os.path.join(folder, "gaussian_cannon_4-01-2026_all_stations.csv"),index=False)
gaussian_part1_summary.to_csv(os.path.join(folder, "gaussian_cannon_4-01-2026_summary.csv"),index=False)
nomagnet_file.to_csv(os.path.join(folder, "no_magnets_station.csv"), index=False)
nomagnet_summary.to_csv(os.path.join(folder, "no_magnets_summary.csv"), index=False)


# ========= PRINT CHECKS ==========
print("=============== Gaussian Groups ===============")
print(
    gaussian_file[["group_start", "group_end", "group"]]
    .drop_duplicates()
    .sort_values(by="group_start")
    .to_string(index=False)
)

print("\n============= Gaussian Trials ==============")
print(gaussian_file.head(40).to_string(index=False))

print("\n============= Gaussian Summary ==============")
print(gaussian_summary.to_string(index=False))

print("\n============== No-Magnets Summary ===============")
print(nomagnet_summary.to_string(index=False))

# ========= OPTIONAL CHECKS ==========
print("\n============= CHECK SPECIAL REGION =============")
special_check = gaussian_file[
    (gaussian_file["run"] >= 135) & (gaussian_file["run"] <= 170)
]
print(special_check.to_string(index=False))

print("\nCounts per Gaussian group and gate:")
print(
    gaussian_file.groupby(["group", "gate"])
    .size()
    .reset_index(name="count")
    .sort_values(by=["group", "gate"])
    .to_string(index=False)
)
