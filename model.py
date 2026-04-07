import numpy as np
from constants import *
import matplotlib.pyplot as plt
from functions import derivative

# All measurements should be in SI units; e.g. m instead of mm

placeholder_dist = 0.1

# ====== PARAMETERS OF MODEL =====

C = -66.20487027569024 / 1000       # Scaling constant for magnetic field / force / moment
h = -3.455433891132098 / 1000       # Distance accounting for offsets in the measurements

# ===== MAGNETIC FIELD OF MAGNET (CURRENTLY ONLY WORKS FOR TWO MAGNETS) =====

def B(x, C, n):
    return ((x + n * mag_thickness + ball_radius) / np.sqrt((x + n * mag_thickness + ball_radius)**2 + mag_radius**2))-((x + ball_radius) / np.sqrt((x + ball_radius)**2 + mag_radius**2))

B = np.vectorize(B)

# ===== INDUCED MAGNETIC MOMENT OF STEEL BALL =====

def m(x, C, n):
    return (C / 2) * B(x, C, n)

m = np.vectorize(m)

# ===== MAGNETIC FORCE BETWEEN BALL AND MAGNET =====

def F_m(x, C, h):
    def B_times_m(x):
        return B(x, C, h) * m(x, C, h)
    return derivative(B_times_m, x, N=10, dx=0.00001)

