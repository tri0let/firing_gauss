#This is for exploring data, writing drafts of analysis, etc.

# from functions import *
from Preliminary import PeakForces
import matplotlib.pyplot as plt
from functions import fastplot, fit
from scipy.signal import find_peaks
import numpy as np


dict = PeakForces('csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', included_runs=[2, 3, 4, 5, 6], plot=False, prominence=0.15)
dict1 = PeakForces('csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', included_runs=[12, 13, 14, 15, 16])
dict0 = PeakForces('csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', included_runs=[51])

peak_forces0 = dict0['forces'][0:8] + dict0['forces'][10:12]

force0 = [np.mean(peak_forces0)]

err0 = [np.std(peak_forces0)/len(peak_forces0)]


force = force0 + dict['mean']

print(force)

force1 = dict1['mean']

err = []

for e, n in zip(dict['error'], dict['noise']):
    err.append(max(e, n))

err = err0 + err

err1 = []

for e, n in zip(dict1['error'], dict1['noise']):
    err1.append(max(e, n))

# for t, r in zip(dict['times'], dict['runs']):
#     print(f'Peak times for run {r}:    {t}')

dist = np.array([0.05, 0.35, 0.67, 1.29, 1.93, 2.57])

# fastplot(dict['raw_data'][0]['time'], dict['raw_data'][0]['force'])

n = 1

def f_1(x, a, h):
    return a / (x - h)**n

def f_2(z, C, a, h):
    return C * (((z - h + 1.27) / np.sqrt((z - h + 1.27)**2 + a**2))-((z - h - 1.27) / np.sqrt((z - h - 1.27)**2 + a**2)))

def f_3(z, C, a, h):
    return C * ((-1.27 - h + z)**2/(a**2 + (-1.27 - h + z)**2)**(3/2) - 1/np.sqrt(a**2 + (-1.27 - h + z)**2) + 1/np.sqrt(a**2 + (1.27 - h + z)**2) - (1.27 - h + z)**2/(a**2 + (1.27 - h + z)**2)**(3/2))

def f_4(z, C, a, h, b):
    return C * ((-b - h + z)**2/(a**2 + (-b - h + z)**2)**(3/2) - 1/np.sqrt(a**2 + (-b - h + z)**2) + 1/np.sqrt(a**2 + (b - h + z)**2) - (b - h + z)**2/(a**2 + (b - h + z)**2)**(3/2))

def f_5(z, C, a, h, b):
    return C * ((-b - h + z)**2/(a**2 + (-b - h + z)**2)**(3/2) - 1/np.sqrt(a**2 + (-b - h + z)**2) + 1/np.sqrt(a**2 + (b - h + z)**2) - (b - h + z)**2/(a**2 + (b - h + z)**2)**(3/2)) * (((z - h + b) / np.sqrt((z - h + b)**2 + a**2))-((z - h - b) / np.sqrt((z - h - b)**2 + a**2)))

def f_6(z, C, h):
    return C * ((-1.27 - h + z)**2/(3.17**2 + (-1.27 - h + z)**2)**(3/2) - 1/np.sqrt(3.17**2 + (-1.27 - h + z)**2) + 1/np.sqrt(3.17**2 + (1.27 - h + z)**2) - (1.27 - h + z)**2/(3.17**2 + (1.27 - h + z)**2)**(3/2)) * (((z - h + 1.27) / np.sqrt((z - h + 1.27)**2 + 3.17**2))-((z - h - 1.27) / np.sqrt((z - h - 1.27)**2 + 3.17**2)))

params, param_err, y_fit, residuals, chi2, chi2_red = fit(model=f_6, x=dist, y=force, yerr=err, label='blue', param_guesses=[13, -2])

print(params)
print(param_err)
print(chi2)

fig, ax = plt.subplots()
ax.errorbar(dist, np.array(force), yerr=err, fmt='o', label='One magnet')
ax.plot(dist, y_fit, label = 'Predicted fit')
# ax.errorbar(np.array(dist), np.array(force1), yerr=err1, fmt='o', label='Two magnets')

# for n in range(1, 12):
#     params, param_err, y_fit, residuals, chi2, chi2_red = fit(model=f_1, x=dist, y=force, yerr=err, label='blue')
#     print(f'n = {n}: reduced chi squared = {chi2_red}')
#     print(f'h: {params['h']}')
#     ax.plot(dist, y_fit, label=f'1/x^{n} fit')

ax.set_xlabel('Distance to magnet surface (m)')
ax.set_ylabel('Force (N)')
ax.legend()
plt.show()