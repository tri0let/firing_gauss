#This is for exploring data, writing drafts of analysis, etc.

# from functions import *
from Preliminary import PeakForces
import matplotlib.pyplot as plt
from functions import fastplot, fit
from scipy.signal import find_peaks
import numpy as np


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