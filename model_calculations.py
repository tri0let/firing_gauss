from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os
from FitParameters import FitParameters
from model import DeltaK
from constants import *
from functions import fit

DEBUG = False

# ========= GET RELEVANT FIT PARAMETERS =========

four_ball_fit_parameters = FitParameters['distance_fit_parameters']['ForcevsDistance2']


# ========= DETERMINE ENERGY LOSS CONSTANT =========

# Use the one-station data to determine the energy loss constant

def ExtractVelocitiesAndDistances(data):
    keys = ['magnets', 'distance_photogates', 'mean_speed_in', 'mean_speed_out', 'sem_speed_in', 'sem_speed_out']

    magnets_velocities = []

    if DEBUG:
        print('\nExtracting velocities and distances...')

    for i, num_stations in enumerate(data['stations']):
        dict = {}
        if num_stations == 1.0:
            for key in keys:
                dict.update({key: data[key][i]})
                if DEBUG:
                    print(f'Added data at index {i} for {num_stations} stations: \n{key}: {data[key][i]}')
        if len(dict) > 0:
            magnets_velocities.append(dict)
    
    return magnets_velocities
            
def AddEnergies(data_list):
    if DEBUG:
        print('\nAdding energies...')

    for i, dict in enumerate(data_list):
        dict['kinetic_energy_in'] = 0.5 * ball_mass * dict['mean_speed_in']**2
        if DEBUG:
            print(f'Added kinetic_energy_in of {0.5 * ball_mass * dict['mean_speed_in']**2} to dictionary {i}')
        dict['sem_kinetic_energy_in'] = ball_mass * dict['mean_speed_in'] * dict['sem_speed_in']
        dict['kinetic_energy_out'] = 0.5 * ball_mass * dict['mean_speed_out']**2
        if DEBUG:
            print(f'Added kinetic_energy_out of {0.5 * ball_mass * dict['mean_speed_out']**2} to dictionary {i}')
        dict['sem_kinetic_energy_out'] = ball_mass * dict['mean_speed_out'] * dict['sem_speed_out']
        dict['change_in_energy'] = dict['kinetic_energy_out'] - dict['kinetic_energy_in']
        if DEBUG:
            print(f'Added change_in_energy of {dict['change_in_energy']} to dictionary {i}')
        dict['sem_change_in_energy'] = np.sqrt(dict['sem_kinetic_energy_in']**2 + dict['sem_kinetic_energy_out']**2)
    return data_list

def AddFitConstant(data_list, fit_results):
    for dict in data_list:
        for result in fit_results:
            if int(result['fit_result']['conditions']['num_magnets']) == int(dict['magnets']):
                dict['C'] = result['fit_result']['params'][0]
                dict['C_err'] = result['fit_result']['param_errors'][0]
    return data_list

def GetEnergyLossConstant(data_list_with_fit):
    def DeltaK_single_variable(num_magnets, D):
        num_magnets = np.array(num_magnets)
        delta_K_list = []
        for num in num_magnets:
            for dict in data_list_with_fit:
                if int(dict['magnets']) == num:
                    C = dict['C']
                    K_init = dict['kinetic_energy_in']
                    initial_distance = dict['distance_photogates'] / 100
                    final_distance = ball_diameter * 3

                    delta_K_list.append(
                        DeltaK(
                            initial_distance=initial_distance,
                            initial_kinetic_energy=K_init,
                            final_distance=final_distance,
                            num_magnets=num,
                            num_balls=4,
                            friction_force=0.00017,
                            C=C,
                            D=D,
                            B=0
                        )
                    )
        if len(delta_K_list) == 0:
            raise ValueError(f'No trial with {num_magnets} magnets found')
        
        if DEBUG:
            print(f'Change in energy for {num_magnets} magnets: {delta_K_list}')

        return np.array(delta_K_list)
    
    x = [2, 4]
    y = []
    yerr = []

    for dict in data_list_with_fit:
        for num_magnets in x:
            if int(dict['magnets']) == num_magnets:
                y.append(dict['change_in_energy'])
                yerr.append(dict['sem_change_in_energy'])
    
    return fit(DeltaK_single_variable, x, y, yerr, label='blue')
            

# ========= LOAD ==========
folder = "csv_data"


g_all = pd.read_csv(os.path.join(folder, "gaussian_cannon_all_stations_summary.csv"))
nm = pd.read_csv(os.path.join(folder, "no_magnets_summary.csv"))

# Rename columns for consistency
if "distance" in g_all.columns:
    g_all = g_all.rename(columns={"distance": "distance_stations"})

# File structure
for df in [ g_all, nm]:
    if "mean_speed" in df.columns:
        df["mean_speed"] = pd.to_numeric(df["mean_speed"], errors="coerce")
    if "sem_speed" in df.columns:
        df["sem_speed"] = pd.to_numeric(df["sem_speed"], errors="coerce")
    if "stations" in df.columns:
        df["stations"] = pd.to_numeric(df["stations"], errors="coerce")
    if "distance_stations" in df.columns:
        df["distance_stations"] = pd.to_numeric(df["distance_stations"], errors="coerce")
    if "distance_photogates" in df.columns:
        df["distance_photogates"] = pd.to_numeric(df["distance_photogates"], errors="coerce")
    if "magnets" in df.columns:
        df["magnets"] = pd.to_numeric(df["magnets"], errors="coerce")
    if "group" in df.columns:
        df["group"] = df["group"].astype(str).str.strip()

# ========= SPLIT ENTRANCE VELOCITY / EXIT VELOCITY ==========
def split(df):
    vin = df[df["gate"] == "1+2"].copy()
    vout = df[df["gate"] == "3+4"].copy()

    # Determine common columns for merging
    common_cols = ["group", "stations", "distance_stations", "magnets"]
    if "distance_photogates" in vin.columns and "distance_photogates" in vout.columns:
        common_cols.append("distance_photogates")

    merged = pd.merge(
        vin,
        vout,
        on=common_cols,
        suffixes=("_in", "_out")
    )
    return merged


g_all = split(g_all)
nm = split(nm)

full = pd.concat([g_all], ignore_index=True)


#======== CALCULATE DISTANCES ==========

def AddProperDistancesAndEnergies(dataframe):
    num_balls = 4
    dataframe['final_distance_to_photogate'] = ball_diameter * (num_balls - 1)
    dataframe['xf_between_stations'] = np.float64(0)

    dataframe['first_run'] = 0
    dataframe['last_run'] = 0

    dataframe['num_balls'] = 4

    for i, group in enumerate(dataframe['group']):
        discard, group = group.split(' ')
        first_run_str, last_run_str = group.split('-')
        dataframe.loc[i, 'first_run'], dataframe.loc[i, 'last_run'] = int(first_run_str), int(last_run_str)


    for i, (stations, first_run, last_run, distance_stations) in enumerate(zip(dataframe['stations'], dataframe['first_run'], dataframe['last_run'], dataframe['distance_stations'])):
        if stations == 2:
            dataframe.loc[i, 'xf_between_stations'] = distance_stations / 100 - ball_diameter
        elif stations == 3 and last_run <= 60:
            dataframe.loc[i, 'xf_between_stations'] = distance_stations / 100 - ball_diameter
        elif stations == 3 and first_run > 120:
            dataframe.loc[i, 'xf_between_stations'] = distance_stations / 100 - ball_diameter - num_balls * mag_thickness
    
    dataframe['K_init'] = 0.5 * ball_mass * dataframe['mean_speed_in']**2
    dataframe['K_final'] = 0.5 * ball_mass * dataframe['mean_speed_out']**2

    return dataframe

def AddOtherConstants(dataframe, fit_results, D, B):
    dataframe['C'] = 0.0

    for i, num_magnets in enumerate(dataframe['magnets']):
        for result in fit_results:
            if int(result['fit_result']['conditions']['num_magnets']) == int(num_magnets):
                dataframe.loc[i, 'C'] = result['fit_result']['params'][0]
        
        dataframe.loc[dataframe['magnets'] == 6, 'C'] = -0.1283425   # Average of 4 balls and 8 balls

    dataframe['friction_force'] = 0.00017

    dataframe['D'] = D

    dataframe['B'] = B

    return dataframe

def AddPredictedEnergy(dataframe):
    x_0 = dataframe['distance_photogates'] / 100
    K_0 = dataframe['K_init']
    interstation = dataframe['xf_between_stations']
    poststation = dataframe['final_distance_to_photogate']
    num_balls = dataframe['num_balls']
    num_magnets = dataframe['magnets']
    friction_force = dataframe['friction_force']
    C = dataframe['C']
    D = dataframe['D']
    B = dataframe['B']

    for i in [1, 2, 3]:
        dataframe[f'K_{i}'] = 0.0

    for i, (stations, x0, k0, inter, post, n_ball, n_mag, f_f, c, d, b, first_run, last_run) in enumerate(zip(dataframe['stations'], x_0, K_0, interstation, poststation, num_balls, num_magnets, friction_force, C, D, B, dataframe['first_run'], dataframe['last_run'])):
        k = 1   
        kinetic_energy = k0
        print(f'Before station 1: {k0}')
        while k < stations:
            if k == 1:
                kinetic_energy += DeltaK(
                    initial_distance=x0,
                    initial_kinetic_energy=kinetic_energy,
                    final_distance=inter,
                    num_magnets=n_mag,
                    friction_force=f_f,
                    C=c,
                    D=d,
                    B=b,
                    num_balls=n_ball
                )
                print(f'After station {k}: {kinetic_energy}')
                dataframe.loc[i, f'K_{k}'] = kinetic_energy
            elif k == 2:
                kinetic_energy += DeltaK(
                    initial_distance=inter,
                    initial_kinetic_energy=kinetic_energy,
                    final_distance=inter,
                    num_magnets=n_mag,
                    friction_force=f_f,
                    C=c,
                    D=d,
                    B=b,
                    num_balls=n_ball
                )
                print(f'After station {k}: {kinetic_energy}')
                dataframe.loc[i, f'K_{k}'] = kinetic_energy
            else:
                raise ValueError('Something went wrong!')
            
            k += 1

        if stations == 1:
            kinetic_energy += DeltaK(
                initial_distance=x0,
                final_distance=post,
                initial_kinetic_energy=kinetic_energy,
                num_balls=n_ball,
                num_magnets=n_mag,
                friction_force=f_f,
                C=c,
                D=d,
                B=b
            )
        
        else:
            kinetic_energy += DeltaK(
                initial_distance=inter,
                final_distance=post,
                initial_kinetic_energy=kinetic_energy,
                num_balls=n_ball,
                num_magnets=n_mag,
                friction_force=f_f,
                C=c,
                D=d,
                B=b
            )
        
        dataframe.loc[i, f'K_{k}'] = kinetic_energy
        
        print(f'After final station: {kinetic_energy}')
        print('=' * 50)

        dataframe.loc[i, 'K_pred'] = kinetic_energy
        
        dataframe.loc[i, 'speed_pred'] = np.sqrt(2 * kinetic_energy / ball_mass)

        dataframe.loc[i, 'predicted_energy_gain'] = kinetic_energy - k0

    return dataframe


#======== MAIN =========

def main():

    data_list = ExtractVelocitiesAndDistances(full)
    data_list = AddEnergies(data_list)
    data_list = AddFitConstant(data_list, four_ball_fit_parameters)

    param_dict, err_dict, y_fit, residuals, chi2, chi2_red = GetEnergyLossConstant(data_list)

    D = param_dict['D']
    B = 0

    friction_force = 0.00017

    new = AddProperDistancesAndEnergies(full)

    new = AddOtherConstants(new, four_ball_fit_parameters, D, B)

    new = AddPredictedEnergy(new)

    print(new[['stations', 'magnets', 'distance_stations', 'mean_speed_out', 'speed_pred', 'K_final', 'K_pred']])


if __name__ == '__main__':
    main()