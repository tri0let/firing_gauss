import os
import math
from Preliminary import extract_trials_from_file,PeakForces, check_requested_runs_exist
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from constants import *

# ========================= UNCERTAINTY =========================
def ConservativeError(data_dict):
    return [float(max(e, n)) for e, n in zip(data_dict['error'], data_dict['noise'])]
# ========================= FIT MODELS =========================

def linear_model(x, m, b):
    return m * x + b

def power_model(x, a, b):
    return a * x**b

n = 1

def f_1(x, a, h):
    x = np.asarray(x, dtype=np.float64)
    a = float(a)
    h = float(h)
    return a / (x - h)**n

def f_2(z, C, a, h):
    z = np.asarray(z, dtype=np.float64)
    C = float(C)
    a = float(a)
    h = float(h)
    return C * (
        ((z - h + 1.27) / np.sqrt((z - h + 1.27)**2 + a**2))
        - ((z - h - 1.27) / np.sqrt((z - h - 1.27)**2 + a**2))
    )

def f_3(z, C, a, h):
    z = np.asarray(z, dtype=np.float64)
    C = float(C)
    a = float(a)
    h = float(h)
    return C * (
        ((-1.27 - h + z)**2 / (a**2 + (-1.27 - h + z)**2)**1.5)
        - 1.0 / np.sqrt(a**2 + (-1.27 - h + z)**2)
        + 1.0 / np.sqrt(a**2 + (1.27 - h + z)**2)
        - ((1.27 - h + z)**2 / (a**2 + (1.27 - h + z)**2)**1.5)
    )

def f_4(z, C, a, h, b):
    z = np.asarray(z, dtype=np.float64)
    C = float(C)
    a = float(a)
    h = float(h)
    b = float(b)
    return C * (
        ((-b - h + z)**2 / (a**2 + (-b - h + z)**2)**1.5)
        - 1.0 / np.sqrt(a**2 + (-b - h + z)**2)
        + 1.0 / np.sqrt(a**2 + (b - h + z)**2)
        - ((b - h + z)**2 / (a**2 + (b - h + z)**2)**1.5)
    )

def f_5(z, C, a, h, b):
    z = np.asarray(z, dtype=np.float64)
    C = float(C)
    a = float(a)
    h = float(h)
    b = float(b)

    term1 = (
        ((-b - h + z)**2 / (a**2 + (-b - h + z)**2)**1.5)
        - 1.0 / np.sqrt(a**2 + (-b - h + z)**2)
        + 1.0 / np.sqrt(a**2 + (b - h + z)**2)
        - ((b - h + z)**2 / (a**2 + (b - h + z)**2)**1.5)
    )

    term2 = (
        ((z - h + b) / np.sqrt((z - h + b)**2 + a**2))
        - ((z - h - b) / np.sqrt((z - h - b)**2 + a**2))
    )

    return C * term1 * term2

def f_6(z, C, h):
    z = np.asarray(z, dtype=np.float64)
    C = float(C)
    h = float(h)
    a = mag_radius
    b = mag_thickness / 2

    term1 = (
        ((-b - h + z)**2 / (a**2 + (-b - h + z)**2)**1.5)
        - 1.0 / np.sqrt(a**2 + (-b - h + z)**2)
        + 1.0 / np.sqrt(a**2 + (b - h + z)**2)
        - ((b - h + z)**2 / (a**2 + (b - h + z)**2)**1.5)
    )

    term2 = (
        ((z - h + b) / np.sqrt((z - h + b)**2 + a**2))
        - ((z - h - b) / np.sqrt((z - h - b)**2 + a**2))
    )

    return C * term1 * term2

def model_fit(z, C, n):
    z = np.asarray(z, dtype=np.float64) / 1000
    C = float(C)
    n = np.asarray(n)
    a = mag_radius
    D = mag_thickness
    R = ball_radius

    term1 = (
        ((n * D + R + z)**2 / (a**2 + (n * D + R + z)**2)**1.5)
        - 1.0 / np.sqrt(a**2 + (n * D + R + z)**2)
        + 1.0 / np.sqrt(a**2 + (R + z)**2)
        - ((R + z)**2 / (a**2 + (R + z)**2)**1.5)
    )

    term2 = (
        ((z + R) / np.sqrt((z + R)**2 + a**2))
        - ((z + n * D + R) / np.sqrt((z + n * D + R)**2 + a**2))
    )

    return C * term1 * term2

def model_fit_offset(z, C, h, n):
    z = np.asarray(z, dtype=np.float64) / 1000
    C = float(C)
    n = np.asarray(n, dtype=int)
    a = mag_radius
    D = mag_thickness
    R = ball_radius

    term1 = (
        ((n * D + R  + h + z)**2 / (a**2 + (n * D + R + h + z)**2)**1.5)
        - 1.0 / np.sqrt(a**2 + (n * D + R + h + z)**2)
        + 1.0 / np.sqrt(a**2 + (R + h + z)**2)
        - ((R + h + z)**2 / (a**2 + (R + h + z)**2)**1.5)
    )

    term2 = (
        ((z + R + h) / np.sqrt((z + R + h)**2 + a**2))
        - ((z + n * D + R + h) / np.sqrt((z + n * D + R + h)**2 + a**2))
    )

    return C * term1 * term2


# ========================= FIT FUNCTION =========================
# ========================= FIT FUNCTION =========================
def FitSituation(x, data_dict, parameters, model='linear'):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(data_dict['mean'], dtype=np.float64)
    yerr = np.asarray(ConservativeError(data_dict), dtype=np.float64)

    if len(x) != len(y):
        raise ValueError(f'Length mismatch: len(x)={len(x)} but len(y)={len(y)}')

    yerr = np.where(yerr <= 0, 1e-12, yerr)

    if model == 'linear':
        fit_function = linear_model
        p0 = [1.0, 0.0]
        bounds = (-np.inf, np.inf)
        param_names = ['m', 'b']

    elif model == 'power':
        fit_function = power_model
        p0 = [1.0, 1.0]   # a = scale, b = exponent
        bounds = ([0, -np.inf], [np.inf, np.inf])  # a > 0
        param_names = ['a', 'b']

    
    elif model == 'f_1':
        fit_function = f_1
        a0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        x_min = np.min(x)
        x_max = np.max(x)
        span = x_max - x_min if x_max > x_min else 1.0
        h0 = x_min - 0.5 * span
        p0 = [a0, h0]
        bounds = (
            [-np.inf, -np.inf],
            [ np.inf, x_min - 1e-6]
        )
        param_names = ['a', 'h']

    elif model == 'f_2':
        fit_function = f_2
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        a0 = 3.17
        h0 = float(x[np.argmax(np.abs(y))]) if len(x) > 0 else 0.0
        p0 = [C0, a0, h0]
        bounds = ([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])
        param_names = ['C', 'a', 'h']

    elif model == 'f_3':
        fit_function = f_3
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        a0 = 3.17
        h0 = float(x[np.argmax(np.abs(y))]) if len(x) > 0 else 0.0
        p0 = [C0, a0, h0]
        bounds = ([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])
        param_names = ['C', 'a', 'h']

    elif model == 'f_4':
        fit_function = f_4
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        a0 = 3.17
        h0 = float(x[np.argmax(np.abs(y))]) if len(x) > 0 else 0.0
        b0 = 1.27
        p0 = [C0, a0, h0, b0]
        bounds = ([-np.inf, 1e-8, -np.inf, 1e-8], [np.inf, np.inf, np.inf, np.inf])
        param_names = ['C', 'a', 'h', 'b']

    elif model == 'f_5':
        fit_function = f_5
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        a0 = 3.17
        h0 = float(x[np.argmax(np.abs(y))]) if len(x) > 0 else 0.0
        b0 = 1.27
        p0 = [C0, a0, h0, b0]
        bounds = ([-np.inf, 1e-8, -np.inf, 1e-8], [np.inf, np.inf, np.inf, np.inf])
        param_names = ['C', 'a', 'h', 'b']

    elif model == 'f_6':
        fit_function = f_6
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        h0 = float(x[np.argmax(np.abs(y))]) if len(x) > 0 else 0.0
        p0 = [C0, h0]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
        param_names = ['C', 'h']

    elif model == 'model_magnets':
        if parameters['num_magnets'] != 'variable':
            raise ValueError(f'Model {model} cannot be used when num_magnets is not variable!')
        z = parameters['distance']
        def model_magnets(n, C):
            return model_fit(z, C, n)
        fit_function = model_magnets
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        p0 = [C0]
        bounds = (-np.inf, np.inf)
        param_names = ['C']

    elif model == 'model_distance':
        if parameters['distance'] != 'variable':
            raise ValueError(f'Model {model} cannot be used when distance is not variable!')
        n = parameters['num_magnets']
        def model_distance(z, C):
            return model_fit(z, C, n)
        fit_function = model_distance
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        p0 = [C0]
        bounds = (-np.inf, np.inf)
        param_names = ['C']

    elif model == 'model_magnets_offset':
        if parameters['num_magnets'] != 'variable':
            raise ValueError(f'Model {model} cannot be used when num_magnets is not variable!')
        z = parameters['distance']
        def model_magnets_offset(n, C, h):
            return model_fit_offset(z, C, h, n)
        fit_function = model_magnets_offset
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        h0 = float(x[np.argmax(np.abs(y))]) if len(x) > 0 else 0.0
        p0 = [C0, h0]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
        param_names = ['C', 'h']

    elif model == 'model_distance_offset':
        if parameters['distance'] != 'variable':
            raise ValueError(f'Model {model} cannot be used when distance is not variable!')
        n = parameters['num_magnets']
        def model_distance_offset(z, C, h):
            return model_fit_offset(z, C, h, n)
        fit_function = model_distance_offset
        C0 = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
        h0 = float(x[np.argmax(np.abs(y))]) if len(x) > 0 else 0.0
        p0 = [C0, h0]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
        param_names = ['C', 'h']

    else:
        raise ValueError(f"Unknown model: {model}")

    popt, pcov = curve_fit(
        fit_function,
        x,
        y,
        sigma=yerr,
        absolute_sigma=True,
        p0=p0,
        bounds=bounds,
        maxfev=200000
    )

    perr = np.sqrt(np.diag(pcov))
    yfit = fit_function(x, *popt)

    chi2 = np.sum(((y - yfit) / yerr) ** 2)
    dof = len(x) - len(popt)
    red_chi2 = chi2 / dof if dof > 0 else np.nan

    return {
        'conditions': parameters,
        'model': model,
        'function': fit_function,
        'params': popt,
        'param_errors': perr,
        'param_names': param_names,
        'yfit': yfit,
        'chi2': chi2,
        'dof': dof,
        'red_chi2': red_chi2
    }
# ========================= PLOT HELPERS =========================
def PlotSituationOnAxis(ax, x, data_dict, label, parameters, model=None, color=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(data_dict['mean'], dtype=np.float64)
    yerr = np.asarray(ConservativeError(data_dict), dtype=np.float64)

    fit_result = None
    if model is not None and len(x) >= 2:
        fit_result = FitSituation(x, data_dict, parameters, model=model)
        xfine = np.linspace(np.min(x), np.max(x), 500)
        yfine = fit_result['function'](xfine, *fit_result['params'])
        ax.plot(
            xfine,
            yfine,
            linewidth=1.8,
            alpha=0.9,
            color=color
        )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='o',
        capsize=4,
        markersize=6,
        linewidth=1.2,
        color=color,
        label=label
    )

    return fit_result

def make_subplot_grid(n_panels, max_cols=3):
    ncols = min(max_cols, max(1, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 4.8 * nrows),
        squeeze=False,
        constrained_layout=True
    )
    return fig, axes.ravel()

def PlotCategoryOnAxis(ax, results_list, title, xlabel, model=None, ylim=None, ylabel=None):
    fit_results = []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    all_x = []

    for i, situation in enumerate(results_list):
        color = color_cycle[i % len(color_cycle)]
        fit_result = PlotSituationOnAxis(
            ax,
            situation['x'],
            situation['data'],
            situation['label'],
            situation['parameters'],
            model=model,
            color=color
        )
        fit_results.append({
            'label': situation['label'],
            'fit_result': fit_result
        })
        all_x.extend(situation['x'])

    legend = ax.legend(
        loc='upper right',
        fontsize=9,
        frameon=True,
        title='Series',
        title_fontsize=9,
        ncol=1
    )
    legend.get_frame().set_alpha(0.9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fit_results


def PlotCategoryFamily(results, keys, family_title, xlabel, ylabel=None, model=None, max_cols=3, show=True):
    if not keys:
        return {}

    keys = sorted(keys, key=lambda s: (len(s), s))
    fig, axes = make_subplot_grid(len(keys), max_cols=max_cols)
    fit_results = {}

    for ax, key in zip(axes, keys):
        fit_results[key] = PlotCategoryOnAxis(
            ax=ax,
            results_list=results[key],
            title=key,
            xlabel=xlabel,
            ylabel=ylabel,
            model=model,
        )

    for ax in axes[len(keys):]:
        ax.set_visible(False)

    fig.suptitle(family_title, fontsize=18)
    if show:
        plt.show()

    return fit_results


# ========================= HELPERS  =========================
def merge_data_dicts(data_dicts):
    merged = {
        'raw_data': [],
        'source_files': [],
        'forces': [],
        'times': [],
        'mean': [],
        'error': [],
        'noise': [],
        'runs': [],
        'num_peaks': []
    }

    for d in data_dicts:
        for key in merged:
            merged[key].extend(d[key])

    return merged


def build_situation_from_datasets(situation, prominence=0.15, plot=False):
    label = situation['label']
    parameters = situation['parameters']

    if 'datasets' in situation:
        x_all = []
        partial_results = []

        for ds in situation['datasets']:
            found_runs, missing_runs = check_requested_runs_exist(ds['file'], ds['runs'])
            if missing_runs:
                raise ValueError(
                    f"{label}: dataset file {ds['file']} is missing requested runs {missing_runs}. "
                    f"Available runs are {found_runs}"
                )

            data = PeakForces(
                ds['file'],
                included_runs=ds['runs'],
                plot=plot,
                prominence=prominence
            )

            if len(ds['x']) != len(data['mean']):
                raise ValueError(
                    f"{label}: dataset file {ds['file']} has len(x)={len(ds['x'])}, "
                    f"requested runs={ds['runs']}, extracted runs={data['runs']}."
                )

            x_all.extend(ds['x'])
            partial_results.append(data)

        merged_data = merge_data_dicts(partial_results)

        return {
            'label': label,
            'parameters': parameters,
            'x': x_all,
            'data': merged_data
        }

    found_runs, missing_runs = check_requested_runs_exist(situation['file'], situation['runs'])
    if missing_runs:
        raise ValueError(
            f"{label}: file {situation['file']} is missing requested runs {missing_runs}. "
            f"Available runs are {found_runs}"
        )

    data = PeakForces(
        situation['file'],
        included_runs=situation['runs'],
        plot=plot,
        prominence=prominence
    )

    if len(situation['x']) != len(data['mean']):
        raise ValueError(
            f"{label}: len(x)={len(situation['x'])}, requested runs={situation['runs']}, "
            f"extracted runs={data['runs']}."
        )

    return {
        'label': label,
        'x': situation['x'],
        'data': data
    }

# ========================= GROUP DEFINITION =========================
GROUPS = {
    'ForcevsDistance1': [
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 2,
                'num_magnets': 1,
            },
            'label': 'Magnet 1',
            'datasets': [
                {
                    'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv',
                    'runs': [2, 3, 4, 5, 6],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                },
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [],
                    'x': []
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 2,
                'num_magnets': 2
            },
            'label': 'Magnet 2',
            'datasets': [
                {
                    'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv',
                    'runs': [12, 13, 14, 15, 16],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                },
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [],
                    'x': []
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 2,
                'num_magnets': 4
            },
            'label': 'Magnet 4',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [6, 7, 8, 9, 10],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 2,
                'num_magnets': 8
            },
            'label': 'Magnet 8',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [21, 22, 23, 24, 25],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 2,
                'num_magnets': 16
            },
            'label': 'Magnet 16',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [36, 37, 38, 39, 40],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ],
        },
    ],

    'ForcevsDistance2': [
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 4,
                'num_magnets': 1
            },
            'label': 'Magnet 1',
            'datasets': [
                {
                    'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv',
                    'runs': [7, 8, 9, 10, 11],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 4,
                'num_magnets': 2
            },
            'label': 'Magnet 2',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [1, 2, 3, 4, 19],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 4,
                'num_magnets': 4
            },
            'label': 'Magnet 4',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [11, 12, 13, 14, 15],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 4,
                'num_magnets': 8
            },
            'label': 'Magnet 8',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [26, 27, 28, 29, 30],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 4,
                'num_magnets': 16
            },
            'label': 'Magnet 16',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [41, 42, 43, 44, 45],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ],
        },
    ],

    'ForcevsDistance3': [
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 6,
                'num_magnets': 2
            },
            'label': 'Magnet 2',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [1, 2, 3, 4, 5],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 6,
                'num_magnets': 4
            },
            'label': 'Magnet 4',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [16, 17, 18, 19, 20],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 6,
                'num_magnets': 8
            },
            'label': 'Magnet 8',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [31, 32, 33, 34, 35],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ]
        },
        {
            'parameters': {
                'distance': 'variable',
                'num_balls': 6,
                'num_magnets': 16
            },
            'label': 'Magnet 16',
            'datasets': [
                {
                    'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv',
                    'runs': [46, 47, 48, 49, 50],
                    'x': [0.35, 0.67, 1.29, 1.93, 2.57]
                }
            ],
        },
    ],

    'ForcevsMagnets1': [
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 2,
                'distance': 0.35   
            },
            'label': 'Distance 1',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [2], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [12], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [6, 21, 36], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 2,
                'distance': 0.67
            },
            'label': 'Distance 2',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [3], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [13], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [7, 22, 37], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 2,
                'distance': 1.29
            },
            'label': 'Distance 3',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [4], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [14], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [8, 23, 38], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
            'num_magnets': 'variable',
            'num_balls': 2,
            'distance': 1.93
            },
            'label': 'Distance 4',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [5], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [15], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [9, 24, 39], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 2,
                'distance': 2.57
            },
            'label': 'Distance 5',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [6], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [16], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [10, 25, 40], 'x': [4, 8, 16]},
            ]
        },
    ],

    'ForcevsMagnets2': [
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 4,
                'distance': 0.35
            },
            
            'label': 'Distance 1',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [7], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [17], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [11, 26, 41], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 4,
                'distance': 0.67
            },
            'label': 'Distance 2',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [8], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [18], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [12, 27, 42], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 4,
                'distance': 1.29
            },
            
            'label': 'Distance 3',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [9], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [19], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [13, 28, 43], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 4,
                'distance': 1.93
            },
            
            'label': 'Distance 4',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [10], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [20], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [14, 29, 44], 'x': [4, 8, 16]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 4,
                'distance': 2.57
            },
            
            'label': 'Distance 5',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [11], 'x': [1]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [21], 'x': [2]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [15, 30, 45], 'x': [4, 8, 16]},
            ]
        },
    ],

    'ForcevsMagnets3': [
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 6,
                'distance': 0.35
            },
            
            'label': 'Distance 1',
            'datasets': [
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [1, 16, 31, 46], 'x': [2, 4, 8, 16]}
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 6,
                'distance': 0.67
            },
            
            'label': 'Distance 2',
            'datasets': [
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [2, 17, 32, 47], 'x': [2, 4, 8, 16]}
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 6,
                'distance': 1.29
            },
            
            'label': 'Distance 3',
            'datasets': [
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [3, 18, 33, 48], 'x': [2, 4, 8, 16]}
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 6,
                'distance': 1.93
            },
            
            'label': 'Distance 4',
            'datasets': [
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [4, 19, 34, 49], 'x': [2, 4, 8, 16]}
            ]
        },
        {
            'parameters': {
                'num_magnets': 'variable',
                'num_balls': 6,
                'distance': 2.57
            },
             
            'label': 'Distance 5',
            'datasets': [
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [5, 20, 35, 50], 'x': [2, 4, 8, 16]}
            ]
        },
    ],

    'ForcevsBalls1': [
        {
            'parameters': {
                'num_magnets': 1,
                'num_balls': 'variable',
                'distance': 0.35
            },
            
            'label': 'Distance 1', 'datasets': [{'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [2, 7], 'x': [2, 4]}]},
        {
            'parameters': {
                'num_magnets': 1,
                'num_balls': 'variable',
                'distance': 0.67
            },
            
            'label': 'Distance 2', 'datasets': [{'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [3, 8], 'x': [2, 4]}]},
        {
            'parameters': {
                'num_magnets': 1,
                'num_balls': 'variable',
                'distance': 1.29
            },
                
            'label': 'Distance 3', 'datasets': [{'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [4, 9], 'x': [2, 4]}]},
        {
            'parameters': {
                'num_magnets': 1,
                'num_balls': 'variable',
                'distance': 1.93
            },
            
            'label': 'Distance 4', 'datasets': [{'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [5, 10], 'x': [2, 4]}]},
        {
            'parameters': {
                'num_magnets': 1,
                'num_balls': 'variable',
                'distance': 2.57
            },
            
            'label': 'Distance 5', 'datasets': [{'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [6, 11], 'x': [2, 4]}]},
    ],

    'ForcevsBalls2': [
        {
            'parameters': {
                'num_magnets': 2,
                'num_balls': 'variable',
                'distance': 0.35
            },
            
            'label': 'Distance 1',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [7], 'x': [2]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [17], 'x': [4]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [1], 'x': [6]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 2,
                'num_balls': 'variable',
                'distance': 0.67
            },
            
            'label': 'Distance 2',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [8], 'x': [2]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [18], 'x': [4]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [2], 'x': [6]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 2,
                'num_balls': 'variable',
                'distance': 1.29
            },
            
            'label': 'Distance 3',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [9], 'x': [2]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [19], 'x': [4]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [3], 'x': [6]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 2,
                'num_balls': 'variable',
                'distance': 1.93
            },
            
            'label': 'Distance 4',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [10], 'x': [2]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [20], 'x': [4]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [4], 'x': [6]},
            ]
        },
        {
            'parameters': {
                'num_magnets': 2,
                'num_balls': 'variable',
                'distance': 2.57
            },
            
            'label': 'Distance 5',
            'datasets': [
                {'file': 'csv_data/03-24-2026_magnetic_pull_data[force_timex11].csv', 'runs': [11], 'x': [2]},
                {'file': 'csv_data/03-24-2026_magnetic_pull_data2[force_timex10].csv', 'runs': [21], 'x': [4]},
                {'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [5], 'x': [6]},
            ]
        },
    ],

    'ForcevsBalls3': [
        {
            'parameters': {
                'num_magnets': 4,
                'num_balls': 'variable',
                'distance': 0.35
            },
            
            'label': 'Distance 1', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [6, 11, 16], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 4,
                'num_balls': 'variable',
                'distance': 0.67
            },
            
            'label': 'Distance 2', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [7, 12, 17], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 4,
                'num_balls': 'variable',
                'distance': 1.29
            },
            
            'label': 'Distance 3', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [8, 13, 18], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 4,
                'num_balls': 'variable',
                'distance': 1.93
            },
            
            'label': 'Distance 4', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [9, 14, 19], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 4,
                'num_balls': 'variable',
                'distance': 2.57
            },
            
            'label': 'Distance 5', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [10, 15, 20], 'x': [2, 4, 6]}]},
    ],

    'ForcevsBalls4': [
        {
            'parameters': {
                'num_magnets': 8,
                'num_balls': 'variable',
                'distance': 0.35
            },
            
            'label': 'Distance 1', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [21, 26, 31], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 8,
                'num_balls': 'variable',
                'distance': 0.67
            },
            
            'label': 'Distance 2', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [22, 27, 32], 'x': [2, 4, 6]}]},
        {
            'parameters': {
            'num_magnets': 8,
            'num_balls': 'variable',
            'distance': 1.29
            },
            
            'label': 'Distance 3', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [23, 28, 33], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 8,
                'num_balls': 'variable',
                'distance': 1.93
            },

            'label': 'Distance 4', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [24, 29, 34], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 8,
                'num_balls': 'variable',
                'distance': 2.57
            },
            
            'label': 'Distance 5', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [25, 30, 35], 'x': [2, 4, 6]}]},
    ],

    'ForcevsBalls5': [
        {
            'parameters': {
                'num_magnets': 16,
                'num_balls': 'variable',
                'distance': 0.35
            },
            
            'label': 'Distance 1', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [36, 41, 46], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 16,
                'num_balls': 'variable',
                'distance': 0.67
            },
            
            'label': 'Distance 2', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [37, 42, 47], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 16,
                'num_balls': 'variable',
                'distance': 1.29
            },
            
            'label': 'Distance 3', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [38, 43, 48], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 16,
                'num_balls': 'variable',
                'distance': 1.93
            },
            
            'label': 'Distance 4', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [39, 44, 49], 'x': [2, 4, 6]}]},
        {
            'parameters': {
                'num_magnets': 16,
                'num_balls': 'variable',
                'distance': 2.57
            },
           
            'label': 'Distance 5', 'datasets': [{'file': 'csv_data/3-25-2026__magnetic_pull_data[force_timex65].csv', 'runs': [40, 45, 50], 'x': [2, 4, 6]}]},
    ],
}


# ========================= VALIDATION =========================
def ValidateGroups(groups):
    for category, situations in groups.items():
        for situation in situations:
            if 'datasets' in situation:
                for ds in situation['datasets']:
                    if len(ds['runs']) != len(ds['x']):
                        raise ValueError(
                            f"{category} -> {situation['label']} -> {ds['file']}: "
                            f"len(runs) = {len(ds['runs'])} but len(x) = {len(ds['x'])}"
                        )
            else:
                if len(situation['runs']) != len(situation['x']):
                    raise ValueError(
                        f"{category} -> {situation['label']}: "
                        f"len(runs) = {len(situation['runs'])} but len(x) = {len(situation['x'])}"
                    )

# ========================= BUILD DATA AUTOMATICALLY =========================
def BuildResults(groups, prominence=0.15, plot=False):
    results = {}
    for category, situations in groups.items():
        results[category] = []

        for situation in situations:
            built = build_situation_from_datasets(
                situation,
                prominence=prominence,
                plot=plot
            )
            results[category].append(built)

    return results

# ========================= SUMMARY TABLES =========================
def summary_dataframe_for_category(category, situations):
    rows = []

    for situation in situations:
        data = situation['data']
        errors = ConservativeError(data)

        for i, (x_val, run, mean_val, err_val, noise_val, n_peaks, source) in enumerate(zip(
            situation['x'],
            data['runs'],
            data['mean'],
            errors,
            data['noise'],
            data['num_peaks'],
            data['source_files']
        )):
            rows.append({
                'Category': category,
                'Series': situation['label'],
                'Point': i + 1,
                'x': round(float(x_val), 3),
                'Run': int(run),
                'Mean Force (N)': round(float(mean_val), 4),
                'Error (N)': round(float(err_val), 4),
                'Noise (N)': round(float(noise_val), 4),
                'Peaks Used': int(n_peaks),
                'Source File': os.path.basename(source)
            })

    return pd.DataFrame(rows)

def fit_summary_dataframe(title, fit_results):
    rows = []

    for item in fit_results:
        label = item['label']
        fit = item['fit_result']

        if fit is None:
            rows.append({
                'Plot Family': title,
                'Series': label,
                'Model': 'None',
                'Reduced χ²': np.nan,
                'Parameters': 'No fit performed'
            })
            continue

        params_text = ', '.join(
            f"{name}={value:.4g} ± {error:.2g}"
            for name, value, error in zip(
                fit['param_names'],
                fit['params'],
                fit['param_errors']
            )
        )

        rows.append({
            'Plot Family': title,
            'Series': label,
            'Model': fit['model'],
            'Reduced χ²': round(float(fit['red_chi2']), 4) if np.isfinite(fit['red_chi2']) else np.nan,
            'Parameters': params_text
        })

    return pd.DataFrame(rows)

def PrintResults(results):
    print('+' + '=' * 100)
    print('DATA SUMMARY TABLES')
    print('=' * 100)

    for category, situations in results.items():
        df = summary_dataframe_for_category(category, situations)
        print(f'\n{category.upper()}')
        print('-' * 100)
        if df.empty:
            print('No data.')
        else:
            print(df.to_string(index=False))

def PrintFitResults(title, fit_results):
    df = fit_summary_dataframe(title, fit_results)
    print('\n' + '=' * 100)
    print(title.upper())
    print('=' * 100)
    print(df.to_string(index=False))
# ========================= MAIN =========================
def main():
    ValidateGroups(GROUPS)

    results = BuildResults(GROUPS, prominence=0.15, plot=False)

    distance_keys = [key for key in results if key.startswith('ForcevsDistance')]
    magnet_keys = [key for key in results if key.startswith('ForcevsMagnets')]
    ball_keys = [key for key in results if key.startswith('ForcevsBalls')]

    distance_fit_results = {}
    for model_name in ['f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'model_distance', 'model_distance_offset']:
        distance_fit_results[model_name] = PlotCategoryFamily(
            results=results,
            family_title=f'Force vs Distance ({model_name})',
            keys=distance_keys,
            xlabel='Distance (mm)',
            ylabel='Force (N)',
            model=model_name,
            max_cols=3
        )
    magnet_fit_results = {}
    for model_name in ['model_magnets', 'model_magnets_offset']:
        magnet_fit_results[model_name] = PlotCategoryFamily(
            results=results,
            family_title=f'Force vs Number of Magnets ({model_name})',
            keys=magnet_keys,
            xlabel='Number of Magnets',
            ylabel='Force (N)',
            model=model_name,
            max_cols=2
        )

    ball_fit_results = PlotCategoryFamily(
        results=results,
        family_title='Force vs Number of Balls',
        keys=ball_keys,
        xlabel='Number of Balls',
        ylabel='Force (N)',
        model='linear',
        max_cols=3
    )

    PrintResults(results)

    for model_name, model_results in distance_fit_results.items():
        for key, fit_result in model_results.items():
            PrintFitResults(f'Force vs Distance ({model_name}) - {key}', fit_result)

    for model_name, model_results in magnet_fit_results.items():
        for key, fit_result in model_results.items():
            PrintFitResults(f'Force vs Number of Magnets ({model_name}) - {key}', fit_result)

    for key, fit_result in ball_fit_results.items():
        PrintFitResults(f'Force vs Number of Balls ({key})', fit_result)
    return results

if __name__ == '__main__':
    RESULTS = main()
