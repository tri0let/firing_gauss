import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#  Colour Dictionary Palette 
COL = {
    "data": "#84A7BD",           # measured data points
    "linear_fit": "#C4AD9D",     # linear model curve
    "quadratic_fit": "#445D48",  # quadratic model curve
    "residuals": "#E2AFA2",      # residual points
    "zero_line": "#91C69F",      # horizontal zero reference
    "grid": "#ECF8F9",           # background grid
    "hist": "#A7C4BC"            # histogram color
}

# ====== Possible function models that I think will be useful for the project ===========
def linear_model(x, m, b):
    return m * x + b
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c
# ================= Random Data Generated To Simulate Measurements ==================
# Normally, like I mentioned for imported data I would use pandas
# but here I generate synthetic data so the code runs without files
def generate_random_data(seed=10):
    np.random.seed(seed)
    x = np.linspace(-5, 5, 35) # parameters used to create the simulated signal
    true_a = 0.5
    true_b = 2.0
    true_c = 3.0
    sigma_true = 3.0                       # assumed measurement noise
    # Gaussian noise added to the underlying signal
    noise = np.random.normal(0, sigma_true, len(x))
    # measured data values
    y = quadratic_model(x, true_a, true_b, true_c) + noise
    # measurement uncertainty assigned to each point
    yerr = np.full_like(y, sigma_true)
    return x, y, yerr

# ======================== Fit Models =================================
# This function performs the fit and also computes residuals and chi-square
def fit(model, x, y, yerr, label):
    popt, pcov = curve_fit(
        model,
        x,
        y,
        sigma=yerr,
        absolute_sigma=True
    )
    # fitted parameters
    params = popt
    # parameter uncertainties from covariance matrix
    param_err = np.sqrt(np.diag(pcov))
    # fitted curve
    y_fit = model(x, *popt)
    # residuals
    residuals = y - y_fit
    # Chi-Squared
    chi2 = np.sum((residuals / yerr) ** 2)
    dof = len(x) - len(popt)
    chi2_red = chi2 / dof
    return params, param_err, y_fit, residuals, chi2, chi2_red

# ========= Noise Analysis ======== 
# Not sure yet if this is the best way to estimate noise
# but this gives some basic quantities from the residuals
def noise_analysis(y_fit, residuals, label):
    # mean residual (checks for systematic offset)
    mean_res = np.mean(residuals)
    # standard deviation of residuals
    std_res = np.std(residuals, ddof=1)
    # RMS residual
    rms_res = np.sqrt(np.mean(residuals**2))
    # signal amplitude estimate (RMS of fitted signal)
    signal_rms = np.sqrt(np.mean(y_fit**2))
    # signal-to-noise ratio
    snr = signal_rms / rms_res if rms_res != 0 else np.inf
    print(f"\nNoise analysis for {label}")
    print("Residual mean:", mean_res)
    print("Residual std :", std_res)
    print("Residual RMS :", rms_res)
    print("Signal RMS   :", signal_rms)
    print("SNR estimate :", snr)

# ================= Plot Data and Residuals =================
def plot_fit(x, y, yerr, y_fit, residuals, title, fit_color, axes):
    ax_fit, ax_res = axes
    ax_fit.errorbar(
        x, y, yerr=yerr,
        fmt='o',
        color=COL["data"],
        capsize=3,
        label="Data"
    )
    ax_fit.plot(
        x,
        y_fit,
        color=fit_color,
        linewidth=2,
        label=title
    )
    ax_fit.set_title(title)
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.grid(True, color=COL["grid"], alpha=0.6)
    ax_fit.legend()
    ax_res.errorbar(
        x, residuals, yerr=yerr,
        fmt='o',
        color=COL["residuals"],
        capsize=3
    )
    ax_res.axhline(
        0,
        linestyle='--',
        color=COL["zero_line"],
        linewidth=1.5
    )
    ax_res.set_title(title + " Residuals")
    ax_res.set_xlabel("x")
    ax_res.set_ylabel("Residuals")
    ax_res.grid(True, color=COL["grid"], alpha=0.6)
# ================= Histograms of Residuals ==============================
# Histograms provide a quick visual check of whether the
# residuals resemble random noise.
def plot_histograms(res_lin, res_quad):
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    axes2[0].hist(
        res_lin,
        bins=10,
        color=COL["hist"],
        edgecolor="black"
    )
    axes2[0].axvline(
        0,
        linestyle='--',
        color=COL["zero_line"]
    )
    axes2[0].set_title("Linear Residual Distribution")
    axes2[0].set_xlabel("Residual")
    axes2[0].set_ylabel("Count")
    axes2[1].hist(
        res_quad,
        bins=10,
        color=COL["hist"],
        edgecolor="black"
    )
    axes2[1].axvline(
        0,
        linestyle='--',
        color=COL["zero_line"]
    )
    axes2[1].set_title("Quadratic Residual Distribution")
    axes2[1].set_xlabel("Residual")
    axes2[1].set_ylabel("Count")
    plt.tight_layout()
    plt.show()
# =============================. Point to Point Noise Estimate ============================
# A quick estimate of measurement fluctuation obtained
# from successive data differences.
def point_to_point_noise(y):
    diff_y = np.diff(y)
    ptp_noise = np.std(diff_y) / np.sqrt(2)
    print("\nEstimated point-to-point noise:", ptp_noise)
# ================= MAIN SCRIPT ==================
x, y, yerr = generate_random_data()
# ----- Linear fit -----
m_lin, m_lin_err, y_fit_lin, res_lin, chi2_lin, chi2_red_lin = fit(
    linear_model, x, y, yerr, "Linear Fit")
# ----- Quadratic fit -----
a_quad, a_quad_err, y_fit_quad, res_quad, chi2_quad, chi2_red_quad =fit(
    quadratic_model, x, y, yerr, "Quadratic Fit"
)
# Noise analysis
noise_analysis(y_fit_lin, res_lin, "Linear Fit")
noise_analysis(y_fit_quad, res_quad, "Quadratic Fit")

# ======= Fit Results Summary =======
print("\n--- Linear Fit ---")
print("chi^2 =", chi2_lin)
print("Reduced chi^2 =", chi2_red_lin)

print("\n--- Quadratic Fit ---")
print("chi^2 =", chi2_quad)
print("Reduced chi^2 =", chi2_red_quad)

# ======================= PLOT DATA AND RESIDUALS ==========================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_fit(x, y, yerr, y_fit_lin, res_lin, "Linear Fit", COL["linear_fit"], (axes[0,0], axes[0,1]))
plot_fit(x, y, yerr, y_fit_quad, res_quad, "Quadratic Fit", COL["quadratic_fit"], (axes[1,0], axes[1,1]))
plt.tight_layout()
plt.show()
# Histogram plots
plot_histograms(res_lin, res_quad)
# Point-to-point noise
point_to_point_noise(y)
