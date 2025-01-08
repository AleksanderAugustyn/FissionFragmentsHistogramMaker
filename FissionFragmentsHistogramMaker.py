"""
This script reads endpoint files and creates histograms of fragment masses with double Gaussian fits.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    """Double Gaussian function for curve fitting"""
    gaussian1 = amp1 * np.exp(-(x - mean1) ** 2 / (2 * sigma1 ** 2))
    gaussian2 = amp2 * np.exp(-(x - mean2) ** 2 / (2 * sigma2 ** 2))
    return gaussian1 + gaussian2


def format_fit_params(amp, mean, sigma):
    """Format fit parameters for display"""
    return f'A = {amp:.3f}\nμ = {mean:.1f}\nσ = {sigma:.1f}'


def process_file(filename):
    """Process the endpoint file and create a histogram of the fragment masses with double Gaussian fit"""
    # Extract Z and N from filename
    parts = filename.split('_')
    Z = int(parts[0])
    N = int(parts[1])
    A = Z + N

    # Read data from file
    volumes = []
    with open(filename) as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) >= 16:
                x = float(columns[15])
                # Add both fragments
                volumes.extend([x * A, (1 - x) * A])

    # Create histogram
    plt.figure(figsize=(12, 7))
    counts, bins, _ = plt.hist(volumes, bins=A // 2, range=(0.25 * A, 0.75 * A), density=True, alpha=0.6,
                               color='skyblue', edgecolor='black', label='Data')

    # Fit double Gaussian to the data
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initial guess for parameters [amp1, mean1, sigma1, amp2, mean2, sigma2]
    # Assuming two peaks roughly symmetric around A/2
    p0 = [
        np.max(counts), A * 0.45, 1,  # First peak
        np.max(counts), A * 0.55, 1  # Second peak
    ]

    try:
        fit_results = curve_fit(double_gaussian, bin_centers, counts, p0=p0)
        popt = fit_results[0]  # Extract the optimal parameters

        # Generate points for smooth curves
        x_fit = np.linspace(0, A, 200)
        y_fit_total = double_gaussian(x_fit, *popt)

        # Individual Gaussian curves
        y_fit1 = popt[0] * np.exp(-(x_fit - popt[1]) ** 2 / (2 * popt[2] ** 2))
        y_fit2 = popt[3] * np.exp(-(x_fit - popt[4]) ** 2 / (2 * popt[5] ** 2))

        # Plot fitted curves
        plt.plot(x_fit, y_fit_total, 'r-', linewidth=2, label='Total fit')
        plt.plot(x_fit, y_fit1, '--', color='black', linewidth=1.5, label='Peak 1')
        plt.plot(x_fit, y_fit2, '--', color='yellow', linewidth=1.5, label='Peak 2')

        # Add fit parameters to plot
        params1 = format_fit_params(popt[0], popt[1], popt[2])
        params2 = format_fit_params(popt[3], popt[4], popt[5])

        # Position text boxes for parameters
        plt.text(0.02, 0.98, 'Peak 1:\n' + params1,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.text(0.98, 0.98, 'Peak 2:\n' + params2,
                 transform=plt.gca().transAxes,
                 horizontalalignment='right',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except RuntimeError as e:
        print(f"Warning: Could not fit double Gaussian curve to the data: {e}")

    plt.xlabel('Fragment Mass (A)')
    plt.ylabel('Probability Density')
    plt.title(f'Fragment Mass Distribution (Z={Z}, N={N}, A={A})')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save plot
    output_filename = filename.replace('Endpoints', 'Histogram')
    output_filename = os.path.splitext(output_filename)[0] + '.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()


# Process the file
input_file = "90_140_20.0_0_1000_FG_0.0_Endpoints.txt"
process_file(input_file)
