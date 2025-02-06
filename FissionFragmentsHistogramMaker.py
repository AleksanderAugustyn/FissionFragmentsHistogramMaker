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


def create_histogram(ax, data, total_value, xlabel, title, color='skyblue'):
    """Create a histogram with double Gaussian fit"""

    # print(total_value)

    counts, bins, _ = ax.hist(data, bins=total_value // 2, range=(0.25 * total_value, 0.75 * total_value),
                              density=True, alpha=0.6, color=color, edgecolor='black', label='Data')

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initial guess for parameters
    p0 = [
        np.max(counts), total_value * 0.45, 1,  # First peak
        np.max(counts), total_value * 0.55, 1  # Second peak
    ]

    fit_success = False
    amplitude_offsets = [0, 0.05, -0.05]  # Offsets to try for amplitude (0%, +5%, -5% of initial guess)

    for amplitude_offset_factor in amplitude_offsets:
        current_p0 = [
            p0[0] * (1 + amplitude_offset_factor), p0[1], p0[2],  # Peak 1: Amp, Mean, Sigma
            p0[3] * (1 + amplitude_offset_factor), p0[4], p0[5]  # Peak 2: Amp, Mean, Sigma
        ]
        try:
            fit_results = curve_fit(double_gaussian, bin_centers, counts, p0=current_p0)
            popt = fit_results[0]
            fit_success = True  # Fit successful in this try
            break  # Exit the loop if fit is successful
        except RuntimeError as e:
            print(f"Warning: Could not fit double Gaussian curve with amplitude offset {amplitude_offset_factor*100:.2f}%: {e}")
            continue  # Try next amplitude offset


    if fit_success:
        x_fit = np.linspace(0.25 * total_value, 0.75 * total_value, 200)
        y_fit_total = double_gaussian(x_fit, *popt)
        y_fit1 = popt[0] * np.exp(-(x_fit - popt[1]) ** 2 / (2 * popt[2] ** 2))
        y_fit2 = popt[3] * np.exp(-(x_fit - popt[4]) ** 2 / (2 * popt[5] ** 2))

        ax.plot(x_fit, y_fit_total, 'r-', linewidth=2, label='Total fit')
        ax.plot(x_fit, y_fit1, '--', color='black', linewidth=1.5, label='Peak 1')
        ax.plot(x_fit, y_fit2, '--', color='yellow', linewidth=1.5, label='Peak 2')

        params1 = format_fit_params(popt[0], popt[1], popt[2])
        params2 = format_fit_params(popt[3], popt[4], popt[5])

        ax.text(0.02, 0.98, 'Peak 1:\n' + params1,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.text(0.98, 0.98, 'Peak 2:\n' + params2,
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        print("Warning: Could not fit double Gaussian curve to the data after multiple attempts.")


    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def process_file(filename):
    """Process the endpoint file and create histograms of fragment masses and charges with double Gaussian fits"""
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
                # Add both fragments for mass distribution
                volumes.extend([x, (1 - x)])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Mass distribution (multiply volumes by A)
    mass_data = [v * A for v in volumes]
    create_histogram(ax1, mass_data, A,
                     'Fragment Mass (A)',
                     f'Fragment Mass Distribution (Z={Z}, N={N}, A={A})')

    # Charge distribution (multiply volumes by Z)
    charge_data = [v * Z for v in volumes]
    create_histogram(ax2, charge_data, Z,
                     'Fragment Charge (Z)',
                     f'Fragment Charge Distribution (Z={Z}, N={N}, A={A})',
                     color='lightgreen')

    plt.tight_layout()

    # Save plot
    output_filename = filename.replace('Endpoints', 'Histogram')
    output_filename = os.path.splitext(output_filename)[0] + '.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Example usage with a single file
    input_file = "90_140_20.0_0_1000_FG_0.0_Endpoints.txt"
    process_file(input_file)
