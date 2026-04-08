import numpy as np
from constants import *
import matplotlib.pyplot as plt
from functions import derivative

# All measurements should be in SI units; e.g. m instead of mm

placeholder_dist = 0.1

# ====== PARAMETERS OF MODEL =====

# C: negative scaling constant for magnetic field / force / moment
# m: number of magnets
# x: positive distance from the surface of the magnet, where
#   0 is ball_radius away from the surface
#   (i.e. x = 0 corresponds to the ball touching the magnet)

# ===== MAGNETIC FIELD OF MAGNET (CURRENTLY ONLY WORKS FOR TWO MAGNETS) =====

def magnetic_field(x, num_magnets):
    if x < 0:
        raise ValueError('x must be a non-negative number!')
    if num_magnets < 0:
        raise ValueError('num_magnets must be a non-negative number!')
    return ((x + num_magnets * mag_thickness + ball_radius) / np.sqrt((x + num_magnets * mag_thickness + ball_radius)**2 + mag_radius**2)) - ((x + ball_radius) / np.sqrt((x + ball_radius)**2 + mag_radius**2))

magnetic_field = np.vectorize(magnetic_field)

# ===== INDUCED MAGNETIC MOMENT OF STEEL BALL =====

def magnetic_moment(x, C, num_magnets):
    if C >= 0:
        raise ValueError('C must be a negative number!')
    return (C / 2) * magnetic_field(x, num_magnets)

magnetic_moment = np.vectorize(magnetic_moment)

# ===== B TIMES m ======

def B_times_m(x, C, num_magnets):
    return magnetic_field(x, num_magnets) * magnetic_moment(x, C, num_magnets)

# ===== MAGNETIC FORCE BETWEEN BALL AND MAGNET =====

def F_m(x, C, num_magnets):
    def differentiable_function(x):
        return B_times_m(x, C, num_magnets)
    return derivative(differentiable_function, x, N=10, dx=0.00001)

# ===== ENERGY GAIN / LOSS =====

def magnetic_work_before(initial_distance, C, num_magnets):
    if initial_distance < 0:
        raise ValueError('initial_distance must be a non_negative number!')
    return C * (magnetic_field(initial_distance, num_magnets)**2 - magnetic_field(0, num_magnets)**2)

def friction_work_before(initial_distance, friction_force):
    if initial_distance < 0:
        raise ValueError('initial_distance must be a non-negative number!')
    if friction_force < 0:
        raise ValueError('friction_force must be a non-negative number!')
    return -friction_force * initial_distance

def magnetic_work_after(final_distance, C, num_magnets, num_balls):
    x_i = ball_diameter * (num_balls - 1)
    if x_i > final_distance:
        raise ValueError('final_distance must not be less than initial position!')
    return C * (magnetic_field(x_i, num_magnets)**2 - magnetic_field(final_distance, num_magnets)**2)

def friction_work_after(final_distance, friction_force, num_balls):
    x_i = ball_diameter * (num_balls - 1)
    if x_i > final_distance:
        raise ValueError('final_distance must not be less than initial position!')
    return -friction_force * (final_distance - x_i)

def energy_loss(initial_kinetic_energy, D, B):
    if D < 0:
        raise ValueError('D must be non-negative!')
    return D * initial_kinetic_energy + B

def DeltaK(initial_distance, initial_kinetic_energy, final_distance, num_magnets, num_balls, friction_force, C, D, B):
    delta_K_before = magnetic_work_before(initial_distance, C, num_magnets) + friction_work_before(initial_distance, friction_force)
    K_collision = initial_kinetic_energy + delta_K_before
    delta_K_after = magnetic_work_after(final_distance, C, num_magnets, num_balls) + friction_work_after(final_distance, friction_force, num_balls)
    E_loss = energy_loss(K_collision, D, B)

    return delta_K_before + delta_K_after - E_loss

