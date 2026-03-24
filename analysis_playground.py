#This is for exploring data, writing drafts of analysis, etc.

# from functions import *
from Preliminary import extract_trials_from_file
import matplotlib.pyplot as plt
from functions import fastplot
from scipy.signal import find_peaks
import numpy as np


#========== Plotting force over distance using March 19th data ==========


force_1_to_11 = extract_trials_from_file('csv_data/03-19-2026_force_measurements_1-11_[force_time_x11].csv')
force_12_to_16 = extract_trials_from_file('csv_data/3-19-2026_force_measurement_12-16_[force_time_x4].csv')

distance1 = [7.52, 7.02, 6.52, 6.02, 5.02]
distance2 = [12.48, 11.98, 11.48, 10.48, 9.48]
distance3 = [10.40, 9.40, 8.40, 7.40, 6.40]

force1 = []
force2 = []
force3 = []

err1 = []
err2 = []
err3 = []

for i in range(1, 6):
    force1.append(force_1_to_11[i]['stats']['mean'])
    err1.append(force_1_to_11[i]['stats']['std'])


for i in range(6, 11):
    force2.append(force_1_to_11[i]['stats']['mean'])
    err2.append(force_1_to_11[i]['stats']['std'])

for i in range(5):
    force3.append(force_12_to_16[i]['stats']['mean'])
    err3.append(force_12_to_16[i]['stats']['std'])


# for i in range(11):
#     plt.plot(force_1_to_11[i]['time'], force_1_to_11[i]['force'], label=f'{force_1_to_11[i]['run']}')
# for i in range(5):
#     plt.plot(force_12_to_16[i]['time'], force_12_to_16[i]['force'], label=f'{force_12_to_16[i]['run']}')
# plt.legend()


fastplot(distance1, force1, yerr=err1, Title='One magnet', xlab='Distance (mm)', ylab='Force (N)')
fastplot(distance2, force2, yerr=err2, Title='Two magnets', xlab='Distance (mm)', ylab='Force (N)')
fastplot(distance3, force3, yerr=err3, Title='Four magnets', xlab='Distance (mm)', ylab='Force (N)')

#Conclusion: the data isn't very useable.


#========== Plotting force over number of magnets using March 23rd data ==========


num_magnets = [1, 2, 4, 8]

pull_force = extract_trials_from_file('csv_data/3-23-2026_magnetic_pull_data[force_time_x5].csv')

force = []
error = []

for trial in pull_force:
    indices, properties = find_peaks(trial['force'], distance=50, prominence=0.2)
    peak_forces = trial['force'][indices]
    if trial['run'] == 2:
        peak_forces = peak_forces[2:]
    force.append(np.mean(peak_forces))
    error.append(np.std(peak_forces)/np.sqrt(len(peak_forces)))

fastplot(num_magnets, force[:4], yerr=error[:4], xlab='Number of magnets', ylab='Force (N)')