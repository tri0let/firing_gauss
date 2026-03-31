import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import fastplot
from scipy.signal import find_peaks

# =========== Settings ========
file_pattern = "csv_data/*force*.csv"
time_column_base = "Time (s)"
force_column_base = "Force (N)"

# Colour palettes
colours = {
    "trial_data": "#84A7BD",
    "mean_curve": "#C4AD9D",
    "noise_curve": "#445D48",
    "zero_line": "#E2AFA2",
}

# Plot configurations
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
})

# This find columns which belongs to each run 
def run_columns(dataframe, base_name):
    run_columns = {}
    pattern = rf"^{re.escape(base_name)}\s+Run\s+#(\d+)$"
    
    for column in dataframe.columns:
        clean_name = column.strip()
        match = re.match(pattern, clean_name)
        
        if match: 
            run_num = int(match.group(1))
            run_columns[run_num] = clean_name
            
    # Sort by run numbers
    run_columns = dict(sorted(run_columns.items()))
    return run_columns

# ============== Compute the nescessary statistics for one run =========
def stats(f_val):
    f_val = np.asarray(f_val, dtype=float)
    f_val = f_val[~np.isnan(f_val)]
    datapoints = len(f_val)
    
    if datapoints == 0:
        return {
            "n_points": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "variance": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "rms": np.nan,
            "sem": np.nan,
            "cv_percent": np.nan,
            "noise_std": np.nan,
            "noise_rms": np.nan,
        }

    mean = np.mean(f_val)
    median = np.median(f_val)
    min = np.min(f_val)
    max = np.max(f_val)
    range = max - min
    rms = np.sqrt(np.mean(f_val ** 2))
    
    if datapoints > 1:
        stdev = np.std(f_val)
        variance = np.var(f_val)
        SEM = stdev / np.sqrt(datapoints)
    else:
        stdev = np.nan
        variance = np.nan
        SEM = np.nan
    
    if mean != 0 and not np.isnan(stdev):
        cv_percent = 100 * stdev / mean              # This is the coeeficient of variation mesures data dispersion relative to the mean
    else: 
        cv_percent = np.nan
        
    # ============= Noise =============
    # Define noise as deviation from trial mean
    noise = f_val - mean
    if datapoints > 1:
        noise_stdev = np.std(noise, ddof=1)
    else:
        noise_stdev = np.nan
    noise_rms = np.sqrt(np.mean(noise ** 2))
    
    return {
        "n_points": datapoints,
        "mean": mean,
        "median": median,
        "std": stdev,
        "variance": variance,
        "min": min,
        "max": max,
        "range": range,
        "rms": rms,
        "sem": SEM,
        "cv_percent": cv_percent,
        "noise_std": noise_stdev, 
        "noise_rms": noise_rms, 
    }
        
      
# =========== Extract all trials in the file ============
def extract_trials_from_file(file_path):

    dataframe = pd.read_csv(file_path)
    dataframe.columns = dataframe.columns.str.strip()

    time_columns = run_columns(dataframe, time_column_base)
    f_columns = run_columns(dataframe, force_column_base)

    # Only keep run numbers that have both time and force columns
    run_numbers = sorted(set(time_columns.keys()) & set(f_columns.keys()))

    if not run_numbers:
        raise ValueError("No matching Time/Force run columns were found.")

    trials = []

    for run_number in run_numbers:
        time_series = pd.to_numeric(dataframe[time_columns[run_number]], errors="coerce")
        f_series = pd.to_numeric(dataframe[f_columns[run_number]], errors="coerce")

        # Put time and force together and drop rows with missing values
        run_dataframe = pd.DataFrame({
            "time": time_series,
            "force": f_series,
        }).dropna()

        run_dataframe = run_dataframe.sort_values("time").reset_index(drop=True)

        # Round time slightly so very tiny floating-point differences
        run_dataframe["time"] = run_dataframe["time"].round(9)

        time_values = run_dataframe["time"].to_numpy(dtype=float)
        f_val = run_dataframe["force"].to_numpy(dtype=float)

        statistics = stats(f_val)
        noise_values = f_val - statistics["mean"]

        one_trial = {
            "run": run_number,
            "time": time_values,
            "force": f_val,
            "noise_within_trial": noise_values,
            "stats": statistics,
        }
        trials.append(one_trial)

    return trials

# ========== Combine Trial ===========
def combine_trials(trials):
    combine_dataframe = None
    
    for trial in trials:
        run_dataframe = pd.DataFrame({
            "time": trial["time"],
            f"f_run_{trial['run']}": trial["force"],
        })
          
        if combine_dataframe is None:
            combine_dataframe = run_dataframe
        else:
            combine_dataframe = pd.merge(
                combine_dataframe,
                run_dataframe,
                on="time",
                how="outer"            
            )
    if combine_dataframe is not None:
        combine_dataframe = combine_dataframe.sort_values("time")
    else: 
        print("Trials could not be combined, combine_dataframe not created!")
        return None
    
    force_column = [
        column for column in combine_dataframe.columns
        if column.startswith("f_run")
    ]
    f_matrix = combine_dataframe[force_column].to_numpy(dtype=float)
    time_val = combine_dataframe["time"].to_numpy(dtype=float)
    
    mean_force = np.nanmean(f_matrix, axis=1)
    stdev_force = np.nanstd(f_matrix, axis=1, ddof=1)
    
    # Noise relative to mean across each trial
    noise_between_trials = f_matrix - mean_force[:, None]
    
    results = {
        "time": time_val,
        "f_matrix": f_matrix,
        "mean_force": mean_force,
        "std_force": stdev_force,
        "noise_between_trials": noise_between_trials,
        "n_trials": len(trials),
    }
    return results

# ============ Print Trial statistics ==========
def print_trial(trials):
    header = "\n" + "=" * 80 + "\nSTATISTICS FOR EACH TRIAL\n" + "=" * 80
    print(header)

    for trial in trials:
        s = trial["stats"]
        output = (
            f"\nRun {trial['run']}\n" + "-" * 40 +
            f"\nNumber of points         = {s['n_points']}"
            f"\nMean force               = {s['mean']:.6f} N"
            f"\nMedian force             = {s['median']:.6f} N"
            f"\nStandard deviation       = {s['std']:.6f} N"
            f"\nVariance                 = {s['variance']:.6f} N^2"
            f"\nMinimum force            = {s['min']:.6f} N"
            f"\nMaximum force            = {s['max']:.6f} N"
            f"\nRange                    = {s['range']:.6f} N"
            f"\nRMS force                = {s['rms']:.6f} N"
            f"\nStandard error of mean   = {s['sem']:.6f} N"
            f"\nCoefficient of variation = {s['cv_percent']:.6f} %"
            f"\nNoise std                = {s['noise_std']:.6f} N"
            f"\nNoise RMS                = {s['noise_rms']:.6f} N"
        )
        print(output)

def make_summary_table(trials):
    """
    Creates a Pandas DataFrame summary. 
    """
    rows = []
    for t in trials:
        s = t["stats"]
        rows.append({
            "Run": t["run"],
            "N points": s["n_points"],
            "Mean (N)": s["mean"],
            "Median (N)": s["median"],
            "Std Dev (N)": s["std"],
            "Variance (N^2)": s["variance"],
            "Min (N)": s["min"],
            "Max (N)": s["max"],
            "Range (N)": s["range"],
            "RMS (N)": s["rms"],
            "SEM (N)": s["sem"],
            "CV (%)": s["cv_percent"],
            "Noise Std (N)": s["noise_std"],
            "Noise RMS (N)": s["noise_rms"],
        })
    return pd.DataFrame(rows)

def print_combined_statistics(combined):
    # Using numpy to handle potential NaN values in the combined dataset
    between_noise = np.nanstd(combined["noise_between_trials"])
    mean_cross_std = np.nanmean(combined["std_force"])

    print("\n" + "=" * 80)
    print("COMBINED STATISTICS ACROSS TRIALS")
    print("=" * 80)
    print(f"Number of trials                  = {combined['n_trials']}")
    print(f"Number of time values             = {len(combined['time'])}")
    print(f"Mean across-trial std vs time      = {mean_cross_std:.6f} N")
    print(f"In between-trial noise std   = {between_noise:.6f} N")

# ============= Plots ===========
def plot_force_for_each_trial(trials, file_label):
    plt.figure(figsize=(8, 5))

    for trial in trials:
        plt.plot(
            trial["time"],
            trial["force"],
            linewidth=1,
            alpha=0.8,
            label=f"Run {trial['run']}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(f"Force vs Time — {file_label}")
    plt.legend()
    plt.tight_layout()


def plot_noise_for_each_trial(trials, file_label):
    plt.figure(figsize=(8, 5))

    for trial in trials:
        plt.plot(
            trial["time"],
            trial["noise_within_trial"],
            linewidth=1,
            alpha=0.8,
            label=f"Run {trial['run']}"
        )

    plt.axhline(0, color=colours["zero_line"], linestyle="--", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Noise (N)")
    plt.title(f"Within-Trial Noise — {file_label}")
    plt.legend()
    plt.tight_layout()


def plot_mean_force_across_trials(combined, file_label):
    valid_mask = ~np.isnan(combined["time"]) & ~np.isnan(combined["mean_force"])

    plt.figure(figsize=(8, 5))
    plt.plot(
        combined["time"][valid_mask],
        combined["mean_force"][valid_mask],
        linewidth=2,
        color=colours["mean_curve"],
        label="Mean force across trials"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(f"Mean Force Across Trials — {file_label}")
    plt.legend()
    plt.tight_layout()


def plot_between_trial_noise(combined, file_label):
    plt.figure(figsize=(8, 5))

    time_values = combined["time"]
    noise_matrix = combined["noise_between_trials"]

    for column_index in range(noise_matrix.shape[1]):
        valid_mask = ~np.isnan(time_values) & ~np.isnan(noise_matrix[:, column_index])

        plt.plot(
            time_values[valid_mask],
            noise_matrix[:, column_index][valid_mask],
            linewidth=1,
            alpha=0.75,
            label=f"Run {column_index + 1}"
        )

    plt.axhline(0, color=colours["zero_line"], linestyle="--", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Noise (N)")
    plt.title(f"Between-Trial Noise — {file_label}")
    plt.legend()
    plt.tight_layout()


# ========================= PEAK REMOVAL =========================

REMOVE_PEAKS = {
    'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv': {
        12: [0],
        14: [2],
        20: [1, 4],
    },
    'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv': {
        16: [9],
    },
    'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv': {
        22: [3, 7],
        23: [2, 5],
        24: [2, 10],
        25: [0],
        39: [2, 8, 10, 12],
        40: [7, 8],
        60: [5],
    }
}


def get_bad_peak_positions(path, run_id):
    file_rules = REMOVE_PEAKS.get(path, {})
    return sorted(set(file_rules.get(run_id, [])))


def remove_bad_peaks(indices, bad_peak_positions):
    if not bad_peak_positions:
        return indices

    mask = np.ones(len(indices), dtype=bool)

    for pos in bad_peak_positions:
        if 0 <= pos < len(indices):
            mask[pos] = False

    return indices[mask]


def check_requested_runs_exist(path, requested_runs):
    if not requested_runs:
        return [], []

    pull_force = extract_trials_from_file(path)
    found_runs = sorted(int(trial['run']) for trial in pull_force)
    missing = [r for r in requested_runs if r not in found_runs]
    return found_runs, missing


# ========================= PEAK FORCE  =========================

def PeakForces(paths,
               distance=50,
               prominence=0.2,
               plot=False,
               included_runs=None):

    runs = []
    num_peaks = []
    forces = []
    times = []
    mean = []
    error = []
    noise = []
    raw_data = []
    source_files = []

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        pull_force = extract_trials_from_file(path)

        for trial in pull_force:
            run_id = int(trial['run'])

            if included_runs is not None and run_id not in included_runs:
                continue

            raw_data.append(trial)
            source_files.append(path)

            indices, _ = find_peaks(
                trial['force'],
                distance=distance,
                prominence=prominence
            )

            if len(indices) == 0:
                raise ValueError(f'No peaks found in file {path}, run {run_id}!')

            bad_peak_positions = get_bad_peak_positions(path, run_id)
            indices = remove_bad_peaks(indices, bad_peak_positions)

            if len(indices) == 0:
                raise ValueError(f'All peaks were removed in file {path}, run {run_id}!')

            peak_forces = trial['force'][indices]
            peak_times = trial['time'][indices]

            if plot:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(trial['time'], trial['force'])
                ax.plot(peak_times, peak_forces, 'o')
                ax.set_title(f"{os.path.basename(path)} | Run #{run_id}")
                ax.set_xlabel('Time')
                ax.set_ylabel('Force')
                ax.grid(alpha=0.25)
                plt.tight_layout()
                plt.show()

            runs.append(run_id)
            num_peaks.append(int(len(indices)))
            forces.append(np.array(peak_forces, dtype=float))
            times.append(np.array(peak_times, dtype=float))

            mean_force = float(np.mean(peak_forces))
            peak_error = float(np.std(peak_forces) / np.sqrt(len(peak_forces))) if len(peak_forces) > 1 else 0.0

            mean.append(mean_force)
            error.append(peak_error)

            noise_index = 0
            i = 0

            while noise_index == 0 and i < max(0, min(indices) - 1):
                if trial['time'][i] >= 5:
                    noise_index = i
                i += 1

            noise_index = i
            noise_floor = float(np.std(trial['force'][0:noise_index + 1]))

            noise.append(noise_floor)

    return {
        'raw_data': raw_data,
        'source_files': source_files,
        'forces': forces,
        'times': times,
        'mean': mean,
        'error': error,
        'noise': noise,
        'runs': runs,
        'num_peaks': num_peaks
    }
    
