import pandas as pd
import re
import math
import os

# ========= FILE PATHS ==========
folder = "csv_data"

file_gaussian_first_station = os.path.join(
    folder,
    "4-01-2026_gaussian_cannon_1st_station[speed_time_between_gatesx40].csv"
)

file_gaussian_all_stations = os.path.join(
    folder,
    "04-02-2026_gaussian_cannon_all_stations[speed_time_between_stationsx191].csv"
)

file_nomagnet = os.path.join(
    folder,
    "3-31-2026_No_Magnets_Station[Speed_Time_Between_Gatesx7].csv"
)

# ========= LOAD FILES ==========
df_gaussian_first_station = pd.read_csv(file_gaussian_first_station)
df_gaussian_all_stations = pd.read_csv(file_gaussian_all_stations)
df_nomagnet = pd.read_csv(file_nomagnet)


# ========= EXTRACT FIRST VALID VALUE PER RUN ==========
def extract_first_values(df, gate, source_name):
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
            "source": source_name,
            "run": run,
            "gate": gate,
            "time": first_time,
            "speed": first_speed
        })

    out = pd.DataFrame(records)
    out = out.sort_values(by=["source", "run", "gate"]).reset_index(drop=True)
    return out


# ========= ASSIGN CONDITIONS ==========
def assign_gaussian_conditions(row):
    """
    Assign stations, distance, magnets, and run grouping from the user's setup.
    Runs 162-171 are removed.
    """
    source = row["source"]
    run = int(row["run"])

    # -------- 4-01-2026 : 1 station --------
    if source == "gaussian_first_station":
        if 1 <= run <= 10:
            return pd.Series([1, 9.979, 2, 1, 10, "Run 1-10"])
        elif 11 <= run <= 20:
            return pd.Series([1, 9.787, 4, 11, 20, "Run 11-20"])
        elif 21 <= run <= 30:
            return pd.Series([1, 10.803, 8, 21, 30, "Run 21-30"])
        elif 31 <= run <= 40:
            return pd.Series([1, 10.925, 6, 31, 40, "Run 31-40"])

    # -------- 04-02-2026 : all stations --------
    elif source == "gaussian_all_stations":
        # Remove this block completely
        if 162 <= run <= 171:
            return pd.Series([None, None, None, None, None, None])

        # 2 stations
        if 1 <= run <= 10:
            return pd.Series([2, 7.05, 2, 1, 10, "Run 1-10"])
        elif 11 <= run <= 20:
            return pd.Series([2, 6.54, 4, 11, 20, "Run 11-20"])
        elif 21 <= run <= 30:
            return pd.Series([2, 6.03, 6, 21, 30, "Run 21-30"])
        elif 61 <= run <= 70:
            return pd.Series([2, 14.61, 2, 61, 70, "Run 61-70"])
        elif 71 <= run <= 80:
            return pd.Series([2, 14.10, 4, 71, 80, "Run 71-80"])
        elif 81 <= run <= 90:
            return pd.Series([2, 13.59, 6, 81, 90, "Run 81-90"])
        elif 91 <= run <= 100:
            return pd.Series([2, 20.80, 2, 91, 100, "Run 91-100"])
        elif 101 <= run <= 110:
            return pd.Series([2, 20.29, 4, 101, 110, "Run 101-110"])
        elif 111 <= run <= 120:
            return pd.Series([2, 19.78, 6, 111, 120, "Run 111-120"])

        # 3 stations
        elif 31 <= run <= 40:
            return pd.Series([3, 7.05, 2, 31, 40, "Run 31-40"])
        elif 41 <= run <= 50:
            return pd.Series([3, 6.54, 4, 41, 50, "Run 41-50"])
        elif 51 <= run <= 60:
            return pd.Series([3, 6.03, 6, 51, 60, "Run 51-60"])
        elif 121 <= run <= 130:
            return pd.Series([3, 21.09, 2, 121, 130, "Run 121-130"])
        elif 131 <= run <= 140:
            return pd.Series([3, 17.58, 2, 131, 140, "Run 131-140"])
        elif 141 <= run <= 151:
            return pd.Series([3, 17.07, 4, 141, 151, "Run 141-151"])
        elif 152 <= run <= 161:
            return pd.Series([3, 16.56, 6, 152, 161, "Run 152-161"])
        elif 172 <= run <= 181:
            return pd.Series([3, 20.58, 4, 172, 181, "Run 172-181"])
        elif 182 <= run <= 191:
            return pd.Series([3, 20.07, 6, 182, 191, "Run 182-191"])

    return pd.Series([None, None, None, None, None, None])


# ========= NO-MAGNET GROUPING ==========
def regroup_nomagnet_trials(df):
    """
    Keep all no-magnet runs together in one group.
    """
    df = df.copy()
    df["stations"] = 0
    df["distance"] = pd.NA
    df["magnets"] = 0
    df["group_start"] = df["run"].min()
    df["group_end"] = df["run"].max()
    df["group"] = "All Runs"
    df = df.sort_values(by=["source", "run", "gate"]).reset_index(drop=True)
    return df


# ========= SUMMARY FUNCTION ==========
def summarize(df):
    """
    Summarize by source, stations, distance, magnets, group, and gate.
    """
    df = df.drop_duplicates(subset=["source", "run", "gate"]).copy()

    summary = (
        df.groupby(
            ["source", "stations", "distance", "magnets", "group_start", "group_end", "group", "gate"],
            sort=True,
            dropna=False
        )
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

    summary = summary.sort_values(
        by=["source", "stations", "group_start", "gate"]
    ).reset_index(drop=True)

    return summary


# ========= PROCESS 4-01-2026 GAUSSIAN FIRST STATION ==========
g1_12_raw = extract_first_values(
    df_gaussian_first_station,
    "1+2",
    "gaussian_first_station"
)
g1_34_raw = extract_first_values(
    df_gaussian_first_station,
    "3+4",
    "gaussian_first_station"
)

gaussian_part1_trials = pd.concat([g1_12_raw, g1_34_raw], ignore_index=True)
gaussian_part1_trials = gaussian_part1_trials.sort_values(
    by=["source", "run", "gate"]
).reset_index(drop=True)

gaussian_part1_trials[
    ["stations", "distance", "magnets", "group_start", "group_end", "group"]
] = gaussian_part1_trials.apply(assign_gaussian_conditions, axis=1)

gaussian_part1_trials = gaussian_part1_trials.dropna(
    subset=["stations", "distance", "magnets", "group"]
).reset_index(drop=True)

gaussian_part1_summary = summarize(gaussian_part1_trials)


# ========= PROCESS 04-02-2026 GAUSSIAN ALL STATIONS ==========
g2_12_raw = extract_first_values(
    df_gaussian_all_stations,
    "1+2",
    "gaussian_all_stations"
)
g2_34_raw = extract_first_values(
    df_gaussian_all_stations,
    "3+4",
    "gaussian_all_stations"
)

gaussian_part2_trials = pd.concat([g2_12_raw, g2_34_raw], ignore_index=True)
gaussian_part2_trials = gaussian_part2_trials.sort_values(
    by=["source", "run", "gate"]
).reset_index(drop=True)

gaussian_part2_trials[
    ["stations", "distance", "magnets", "group_start", "group_end", "group"]
] = gaussian_part2_trials.apply(assign_gaussian_conditions, axis=1)

gaussian_part2_trials = gaussian_part2_trials.dropna(
    subset=["stations", "distance", "magnets", "group"]
).reset_index(drop=True)

gaussian_part2_summary = summarize(gaussian_part2_trials)


# =========  GAUSSIAN SUMMARIES ==========
gaussian_all_trials = pd.concat(
    [gaussian_part1_trials, gaussian_part2_trials],
    ignore_index=True
).sort_values(by=["source", "stations", "group_start", "run", "gate"]).reset_index(drop=True)

gaussian_all_summary = summarize(gaussian_all_trials)


# ========= PROCESS NO-MAGNET FILE ==========
nm_12_raw = extract_first_values(df_nomagnet, "1+2", "no_magnets")
nm_34_raw = extract_first_values(df_nomagnet, "3+4", "no_magnets")

nomagnet_trials = pd.concat([nm_12_raw, nm_34_raw], ignore_index=True)
nomagnet_trials = nomagnet_trials.sort_values(
    by=["source", "run", "gate"]
).reset_index(drop=True)

nomagnet_trials = regroup_nomagnet_trials(nomagnet_trials)
nomagnet_summary = summarize(nomagnet_trials)


# ========= SUMMARY COLUMNS ==========
gaussian_part1_summary = gaussian_part1_summary[
    [
        "source", "stations", "distance", "magnets",
        "group", "gate",
        "n_speed", "mean_speed", "std_speed", "sem_speed",
        "n_time", "mean_time", "std_time", "sem_time"
    ]
].copy()

gaussian_all_summary = gaussian_all_summary[
    [
        "source", "stations", "distance", "magnets",
        "group", "gate",
        "n_speed", "mean_speed", "std_speed", "sem_speed",
        "n_time", "mean_time", "std_time", "sem_time"
    ]
].copy()

nomagnet_summary = nomagnet_summary[
    [
        "source", "stations", "distance", "magnets",
        "group", "gate",
        "n_speed", "mean_speed", "std_speed", "sem_speed",
        "n_time", "mean_time", "std_time", "sem_time"
    ]
].copy()


# ========= FINAL DATAFRAMES  ==========
gaussian_all_summary.to_csv(os.path.join(folder, "gaussian_cannon_all_stations_summary.csv"), index=False)
nomagnet_summary.to_csv(os.path.join(folder, "no_magnets_summary.csv"), index=False)


# ========= FILE STRUCTURE ==========
for df in [gaussian_all_summary, nomagnet_summary]:
    df["mean_speed"] = pd.to_numeric(df["mean_speed"], errors="coerce")
    df["sem_speed"] = pd.to_numeric(df["sem_speed"], errors="coerce")
    df["stations"] = pd.to_numeric(df["stations"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["magnets"] = pd.to_numeric(df["magnets"], errors="coerce")
    df["group"] = df["group"].astype(str).str.strip()

