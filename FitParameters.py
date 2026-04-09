import os
import math
from Preliminary import extract_trials_from_file,PeakForces, check_requested_runs_exist
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from constants import *
from Analysis_playgroud_magnetic_force_without_first_distance import (
    model_fit,
    ValidateGroups,
    BuildResults,
    PlotCategoryFamily,
    PrintFitResults,
    PrintResults,
    GROUPS
)

# ========================= MAIN =========================
def main():
    ValidateGroups(GROUPS)

    results = BuildResults(GROUPS, prominence=0.15, plot=False)

    distance_keys = [key for key in results if key.startswith('ForcevsDistance')]
    magnet_keys = [key for key in results if key.startswith('ForcevsMagnets')]


    distance_fit_results = PlotCategoryFamily(
        results=results,
        family_title=f'Force vs Distance (model_distance)',
        keys=distance_keys,
        xlabel='Distance (mm)',
        ylabel='Force (N)',
        model='model_distance',
        max_cols=3,
        show=False
        )
    
    magnet_fit_results = PlotCategoryFamily(
        results=results,
        family_title=f'Force vs Number of Magnets (model_magnets)',
        keys=magnet_keys,
        xlabel='Number of Magnets',
        ylabel='Force (N)',
        model='model_magnets',
        max_cols=2,
        show=False
    )

    # PrintResults(results)

    # for model_name, model_results in distance_fit_results.items():
    #     for key, fit_result in model_results.items():
    #         PrintFitResults(f'Force vs Distance ({model_name}) - {key}', fit_result)

    # for model_name, model_results in magnet_fit_results.items():
    #     for key, fit_result in model_results.items():
    #         PrintFitResults(f'Force vs Number of Magnets ({model_name}) - {key}', fit_result)

    return {'distance_fit_parameters': distance_fit_results, 'magnet_fit_parameters': magnet_fit_results}

FitParameters = main()
