"""
This script creates a combined plot of Z histograms from multiple input files.
"""
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.optimize import curve_fit

from FissionFragmentsHistogramMaker import double_gaussian, format_fit_params


def create_z_histogram(ax, data, Z, N, color='lightgreen') -> ndarray | Iterable | int | float:
    """Create a histogram with double Gaussian fit with fixed axes"""
    counts, bins, _ = ax.hist(data, bins=40, range=(25, 65),
                              density=True, alpha=0.6, color=color,
                              edgecolor='black', label='Data')

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initial guess for parameters
    p0 = [
        np.max(counts), 0.45 * Z, 1,  # First peak
        np.max(counts), 0.55 * Z, 1  # Second peak
    ]

    try:
        fit_results = curve_fit(double_gaussian, bin_centers, counts, p0=p0)
        popt = fit_results[0]

        x_fit = np.linspace(25, 65, 200)
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

        ax.set_xlabel('Fragment Charge (Z)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Fragment Charge Distribution (Z={Z}, N={N})')
        ax.set_ylim(0, 0.3)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return popt

    except RuntimeError as e:
        print(f"Warning: Could not fit double Gaussian curve to the data: {e}")
        ax.set_xlabel('Fragment Charge (Z)')
        ax.set_ylabel('Probability Density') 
        ax.set_title(f'Fragment Charge Distribution (Z={Z}, N={N})')
        ax.set_ylim(0, 0.3)
        ax.grid(True, alpha=0.3)
        ax.legend()
        return None


def process_multiple_files(filenames):
    """Process multiple endpoint files and create a combined plot of Z histograms"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
    reference_peak = None  # Store position of second peak for Z=94

    for idx, filename in enumerate(filenames):
        # Extract Z and N from filename
        parts = filename.split('_')
        Z = int(parts[0])
        N = int(parts[1])

        # Read data from file
        volumes = []
        with open(filename) as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) >= 16:
                    x = float(columns[15])
                    # Add both fragments for charge distribution
                    volumes.extend([x, (1 - x)])

        # Charge distribution (multiply volumes by Z)
        charge_data = [v * Z for v in volumes]
        fit_params = create_z_histogram(axes[idx], charge_data, Z, N)

        # Store the position of second peak for Z=94, N=146
        if Z == 94 and N == 146 and fit_params is not None:
            reference_peak = fit_params[4]  # Position of second peak

    # Add vertical reference line if we found the peak
    if reference_peak is not None:
        for ax in axes:
            ax.axvline(x=reference_peak, color='black', linestyle=':', linewidth=1.5,
                       label='Z=94 Peak 2' if ax == axes[0] else "")
            ax.legend()

    plt.tight_layout()
    plt.savefig('combined_z_histograms.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


# Example usage with 4 files
input_files = [
    "98_152_20.0_0_1000_FG_0.0_Endpoints.txt",
    "94_146_20.0_0_1000_FG_0.0_Endpoints.txt",
    "92_144_20.0_0_1000_FG_0.0_Endpoints.txt",
    "90_140_20.0_0_1000_FG_0.0_Endpoints.txt"
]

process_multiple_files(input_files)
