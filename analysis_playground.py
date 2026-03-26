#This is for exploring data, writing drafts of analysis, etc.

# from functions import *
from Preliminary import extract_trials_from_file
import matplotlib.pyplot as plt
from functions import fastplot, fit
from scipy.signal import find_peaks
import numpy as np
from inspect import signature


# #========== Plotting force over distance using March 19th data ==========


# force_1_to_11 = extract_trials_from_file('csv_data/03-19-2026_force_measurements_1-11_[force_time_x11].csv')
# force_12_to_16 = extract_trials_from_file('csv_data/3-19-2026_force_measurement_12-16_[force_time_x4].csv')

# distance1 = [7.52, 7.02, 6.52, 6.02, 5.02]
# distance2 = [12.48, 11.98, 11.48, 10.48, 9.48]
# distance3 = [10.40, 9.40, 8.40, 7.40, 6.40]

# force1 = []
# force2 = []
# force3 = []

# err1 = []
# err2 = []
# err3 = []

# for i in range(1, 6):
#     force1.append(force_1_to_11[i]['stats']['mean'])
#     err1.append(force_1_to_11[i]['stats']['std'])


# for i in range(6, 11):
#     force2.append(force_1_to_11[i]['stats']['mean'])
#     err2.append(force_1_to_11[i]['stats']['std'])

# for i in range(5):
#     force3.append(force_12_to_16[i]['stats']['mean'])
#     err3.append(force_12_to_16[i]['stats']['std'])


# # for i in range(11):
# #     plt.plot(force_1_to_11[i]['time'], force_1_to_11[i]['force'], label=f'{force_1_to_11[i]['run']}')
# # for i in range(5):
# #     plt.plot(force_12_to_16[i]['time'], force_12_to_16[i]['force'], label=f'{force_12_to_16[i]['run']}')
# # plt.legend()


# fastplot(distance1, force1, yerr=err1, Title='One magnet', xlab='Distance (mm)', ylab='Force (N)')
# fastplot(distance2, force2, yerr=err2, Title='Two magnets', xlab='Distance (mm)', ylab='Force (N)')
# fastplot(distance3, force3, yerr=err3, Title='Four magnets', xlab='Distance (mm)', ylab='Force (N)')

# #Conclusion: the data isn't very useable.


# #========== Plotting force over number of magnets using March 23rd data ==========


# num_magnets = [1, 2, 4, 8]

# pull_force = extract_trials_from_file('csv_data/3-23-2026_magnetic_pull_data[force_time_x5].csv')

# force = []
# error = []

# for trial in pull_force:
#     indices, properties = find_peaks(trial['force'], distance=50, prominence=0.2)
#     peak_forces = trial['force'][indices]
#     if trial['run'] == 2:
#         peak_forces = peak_forces[2:]
#     force.append(np.mean(peak_forces))
#     error.append(np.std(peak_forces)/np.sqrt(len(peak_forces)))

# fastplot(num_magnets, force[:4], yerr=error[:4], xlab='Number of magnets', ylab='Force (N)')


def PeakForces(paths: list[str] | str, distance: int=50, prominence: float=0.2, plot: bool=False, included_runs: list[int] | None=None) -> dict:
    runs = []
    num_peaks = []
    forces = []
    times = []
    mean = []
    error = []
    noise = []
    raw_data = []

    if not isinstance(paths, list):
        paths = [paths]
    
    for path in paths:
        pull_force = extract_trials_from_file(path)

        for trial in pull_force:

            if included_runs is not None:
                if trial['run'] not in included_runs:
                    continue

            raw_data.append(trial)

            indices, properties = find_peaks(trial['force'], distance=distance, prominence=prominence)
            if len(indices) == 0:
                raise ValueError('No peaks found!')
            peak_forces = trial['force'][indices]
            peak_times = trial['time'][indices]

            if plot:
                fastplot(trial['time'], trial['force'], Title=f"Run #{trial['run']}")

            runs.append(trial['run'])
            num_peaks.append(len(indices))
            forces.append(peak_forces)
            times.append(peak_times)
            mean.append(np.mean(peak_forces))
            error.append(np.std(peak_forces)/np.sqrt(len(peak_forces)))

            noise_index = 0
            i = 0

            while noise_index == 0 and i < min(indices) - 1:
                if trial['time'][i] >= 5:
                    noise_index = i
                i += 1

            noise_index = i

            noise_floor = np.std(trial['force'][0:noise_index + 1])

            if plot:
                fastplot(trial['time'][0:noise_index + 1], trial['force'][0:noise_index + 1], Title=f"Run #{trial['run']} five second noise")

            noise.append(noise_floor)

        return {'raw_data': raw_data, 'forces': forces, 'times': times, 'mean': mean, 'error': error, 'noise': noise, 'runs': runs, 'num_peaks': num_peaks}
    


dict = PeakForces('csv_data/03-24-2026_magneti_pull_data[force_timex11].csv', included_runs=[2, 3, 4, 5, 6], plot=False, prominence=0.15)
dict1 = PeakForces('csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', included_runs=[12, 13, 14, 15, 16])

force = dict['mean']

force1 = dict1['mean']


err = []

for e, n in zip(dict['error'], dict['noise']):
    err.append(max(e, n))

err1 = []

for e, n in zip(dict1['error'], dict1['noise']):
    err1.append(max(e, n))

# for t, r in zip(dict['times'], dict['runs']):
#     print(f'Peak times for run {r}:    {t}')

dist = np.array([0.35, 0.67, 1.29, 1.93, 2.57])/1000

# fastplot(dict['raw_data'][0]['time'], dict['raw_data'][0]['force'])

n = 1

def f_1(x, a, h):
    return a / (x - h)**n

params, param_err, y_fit, residuals, chi2, chi2_red = fit(model=f_1, x=dist, y=force, yerr=err, label='blue')

print(params)
print(param_err)
print(chi2)

fig, ax = plt.subplots()
ax.errorbar(np.array(dist), np.array(force), yerr=err, fmt='o', label='One magnet')
ax.errorbar(np.array(dist), np.array(force1), yerr=err1, fmt='o', label='Two magnets')



for n in range(1, 12):
    params, param_err, y_fit, residuals, chi2, chi2_red = fit(model=f_1, x=dist, y=force, yerr=err, label='blue')
    print(f'n = {n}: chi2 = {chi2}')
    ax.plot(dist, y_fit, label=f'1/x^{n} fit')

ax.set_xlabel('Distance to magnet surface (m)')
ax.set_ylabel('Force (N)')
ax.legend()
plt.show()