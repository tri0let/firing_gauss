import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# FILE PATH
csv_file_path = Path("csv_data/3-31-2026_No_Magnets_Station[Speed_Time_Between_Gatesx7].csv")


""" RUN INFORMATION
start_position_cm = location of first measured photogate section
section_distance_cm = distance between the two measured gate sections"""
run_information = {
    2: {"start_position_cm": 22.40, "section_distance_cm": 15.05},
    3: {"start_position_cm": 22.40, "section_distance_cm": 28.15},
    4: {"start_position_cm": 22.40, "section_distance_cm": 29.15},
    5: {"start_position_cm": 22.40, "section_distance_cm": 42.20},
    6: {"start_position_cm": 22.40, "section_distance_cm": 43.25},
    7: {"start_position_cm": 22.40, "section_distance_cm": 56.35},
    8: {"start_position_cm": 22.40, "section_distance_cm": 57.40},
}

# LOAD DATA
dataframe = pd.read_csv(csv_file_path)


""" TRIAL TABLE
v1 = speed from photogates 1 and 2
v2 = speed from photogates 3 and 4
t1 = time from photogates 1 and 2
t2 = time from photogates 3 and 4""" 
trial_rows = []

for run_number, run_values in run_information.items():

    speed_12_column = f"Speed Between Gates, Ch 1+2 (m/s) Run #{run_number}"
    speed_34_column = f"Speed Between Gates, Ch 3+4 (m/s) Run #{run_number}"
    time_12_column = f"Time Between Gates, Ch 1+2 (s) Run #{run_number}"
    time_34_column = f"Time Between Gates, Ch 3+4 (s) Run #{run_number}"

    speed_12 = dataframe[speed_12_column].dropna().to_numpy()
    speed_34 = dataframe[speed_34_column].dropna().to_numpy()
    time_12 = dataframe[time_12_column].dropna().to_numpy()
    time_34 = dataframe[time_34_column].dropna().to_numpy()

    trial_count = min(len(speed_12), len(speed_34), len(time_12), len(time_34))

    speed_12 = speed_12[:trial_count]
    speed_34 = speed_34[:trial_count]
    time_12 = time_12[:trial_count]
    time_34 = time_34[:trial_count]

    start_position_m = run_values["start_position_cm"] / 100.0
    section_distance_m = run_values["section_distance_cm"] / 100.0

    delta_time_s = time_34 - time_12
    delta_speed_m_per_s = speed_34 - speed_12
    mean_speed_m_per_s = (speed_12 + speed_34) / 2.0

    acceleration_m_per_s2 = (speed_34**2 - speed_12**2) / (2.0 * section_distance_m)
    friction_m_per_s2 = -acceleration_m_per_s2

    for trial_index in range(trial_count):
        trial_rows.append({
            "run": run_number,
            "trial": trial_index + 1,
            "start_position_m": start_position_m,
            "section_distance_m": section_distance_m,
            "v1_m_per_s": speed_12[trial_index],
            "v2_m_per_s": speed_34[trial_index],
            "t1_s": time_12[trial_index],
            "t2_s": time_34[trial_index],
            "delta_time_s": delta_time_s[trial_index],
            "delta_speed_m_per_s": delta_speed_m_per_s[trial_index],
            "mean_speed_m_per_s": mean_speed_m_per_s[trial_index],
            "friction_m_per_s2": friction_m_per_s2[trial_index],
        })

trial_data = pd.DataFrame(trial_rows)
trial_data = trial_data.sort_values(by=["run", "trial"]).reset_index(drop=True)

print("\nTRIAL DATA")
print(trial_data.round(5).to_string(index=False))

# SUMMARY BY RUN

run_summary = (
    trial_data.groupby("run")
    .agg(
        trials=("trial", "count"),
        start_position_m=("start_position_m", "first"),
        section_distance_m=("section_distance_m", "first"),

        mean_v1_m_per_s=("v1_m_per_s", "mean"),
        std_v1_m_per_s=("v1_m_per_s", "std"),

        mean_v2_m_per_s=("v2_m_per_s", "mean"),
        std_v2_m_per_s=("v2_m_per_s", "std"),

        mean_delta_time_s=("delta_time_s", "mean"),
        std_delta_time_s=("delta_time_s", "std"),

        mean_delta_speed_m_per_s=("delta_speed_m_per_s", "mean"),
        std_delta_speed_m_per_s=("delta_speed_m_per_s", "std"),

        mean_friction_m_per_s2=("friction_m_per_s2", "mean"),
        std_friction_m_per_s2=("friction_m_per_s2", "std"),
    )
    .reset_index()
)

run_summary["sem_friction_m_per_s2"] = (
    run_summary["std_friction_m_per_s2"] / np.sqrt(run_summary["trials"])
)

run_summary = run_summary.sort_values(by="run").reset_index(drop=True)
run_summary = run_summary.round(5)

print("\nRUN SUMMARY")
print(run_summary.to_string(index=False))

# OVERALL SUMMARY
overall_summary = pd.DataFrame([{
    "total_trials": len(trial_data),
    "mean_v1_m_per_s": trial_data["v1_m_per_s"].mean(),
    "std_v1_m_per_s": trial_data["v1_m_per_s"].std(),
    "mean_v2_m_per_s": trial_data["v2_m_per_s"].mean(),
    "std_v2_m_per_s": trial_data["v2_m_per_s"].std(),
    "mean_delta_time_s": trial_data["delta_time_s"].mean(),
    "std_delta_time_s": trial_data["delta_time_s"].std(),
    "mean_delta_speed_m_per_s": trial_data["delta_speed_m_per_s"].mean(),
    "std_delta_speed_m_per_s": trial_data["delta_speed_m_per_s"].std(),
    "mean_friction_m_per_s2": trial_data["friction_m_per_s2"].mean(),
    "std_friction_m_per_s2": trial_data["friction_m_per_s2"].std(),
    "sem_friction_m_per_s2": trial_data["friction_m_per_s2"].std() / np.sqrt(len(trial_data)),
}]).round(5)

print("\nOVERALL SUMMARY")
print(overall_summary.to_string(index=False))


# PLOT: FRICTION PER RUN + OVERALL
plt.figure(figsize=(8, 5))

plt.errorbar(
    run_summary["section_distance_m"],
    run_summary["mean_friction_m_per_s2"],
    yerr=run_summary["sem_friction_m_per_s2"],
    fmt="o",
    capsize=4,
    label="Mean friction for each run"
)

overall_mean_friction = overall_summary.loc[0, "mean_friction_m_per_s2"]

plt.axhline(
    overall_mean_friction,
    linestyle="--",
    label=f"Overall mean friction = {overall_mean_friction:.5f}"
)

plt.xlabel("Distance between photogate sections (m)")
plt.ylabel("Friction acceleration (m/s^2)")
plt.title("Average friction by run and overall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""# OPTIONAL DRAG CHECK
speed_values = trial_data["mean_speed_m_per_s"].to_numpy()
friction_values = trial_data["friction_m_per_s2"].to_numpy()

constant_fit = np.polyfit(speed_values, friction_values, 0)
linear_fit = np.polyfit(speed_values, friction_values, 1)
quadratic_fit = np.polyfit(speed_values, friction_values, 2)


def sse(y, y_fit):
    return np.sum((y - y_fit) ** 2)


print("\nDRAG CHECK")
print("constant:", sse(friction_values, np.polyval(constant_fit, speed_values)))
print("linear:", sse(friction_values, np.polyval(linear_fit, speed_values)))
print("quadratic:", sse(friction_values, np.polyval(quadratic_fit, speed_values)))

speed_smooth = np.linspace(np.min(speed_values), np.max(speed_values), 300)

# PLOT 2: SUBPLOTS FOR EACH MODEL 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models = [
    ("Constant", constant_fit),
    ("Linear", linear_fit),
    ("Quadratic", quadratic_fit)
]

for ax, (name, fit) in zip(axes, models):
    ax.scatter(speed_values, friction_values)

    y_fit = np.polyval(fit, speed_smooth)
    ax.plot(speed_smooth, y_fit)

    sse_val = sse(friction_values, np.polyval(fit, speed_values))

    ax.set_title(f"{name} fit\nSSE = {sse_val:.3e}")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Friction (m/s²)")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()"""