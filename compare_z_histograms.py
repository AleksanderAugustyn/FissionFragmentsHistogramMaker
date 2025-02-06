"""
This script creates a combined plot of Z histograms from multiple input files.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from FissionFragmentsHistogramMaker import double_gaussian, format_fit_params

# Dictionary mapping atomic numbers to element names
ELEMENTS = {
    88: "Ra",
    90: "Th",
    92: "U",
    94: "Pu",
    96: "Cm",
    98: "Cf",
    100: "Fm",
    102: "No",
    104: "Rf",
    106: "Sg",
    108: "Hs",
    110: "Ds",
    112: "Cn",
    114: "Fl",
    116: "Lv",
    118: "Og"
}


def create_z_histogram(ax, data, Z, N, color='lightgreen') -> Optional[np.ndarray]:
    """Create a histogram with double Gaussian fit with fixed axes"""
    counts, bins, _ = ax.hist(data, bins=40, range=(25, 65),
                              density=True, alpha=0.6, color=color,
                              edgecolor='black', label='Data')

    # Calculate sum of probability densities
    bin_width = bins[1] - bins[0]
    prob_density_sum = np.sum(counts) * bin_width
    print(f"Sum of probability densities: {prob_density_sum:.3f}")

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initial guess for parameters
    p0 = [
        np.max(counts), 0.45 * Z, 1,  # First peak
        np.max(counts), 0.55 * Z, 1  # Second peak
    ]

    fit_success = False
    amplitude_offsets = [0, 0.05, -0.05, 0.1, -0.1]  # Offsets to try for amplitude (0%, +5%, -5% of initial guess)
    mean_offsets = [0, 0.5, -0.5]  # Offsets to try for mean (0%, +5%, -5% of initial guess)

    for amplitude_offset_factor in amplitude_offsets:
        for mean_offset_factor in mean_offsets:
            current_p0 = [
                p0[0] * (1 + amplitude_offset_factor), p0[1] + mean_offset_factor, p0[2],  # Peak 1: Amp, Mean, Sigma
                p0[3] * (1 + amplitude_offset_factor), p0[4] + mean_offset_factor, p0[5]  # Peak 2: Amp, Mean, Sigma
            ]
            try:
                fit_results = curve_fit(double_gaussian, bin_centers, counts, p0=current_p0)
                popt = fit_results[0]
                fit_success = True  # Fit successful in this try
                break  # Exit the loop if fit is successful
            except RuntimeError as e:
                print(f"Warning: Could not fit double Gaussian curve with amplitude offset {amplitude_offset_factor * 100:.2f}%: {e}")
                continue

    if fit_success:
        x_fit = np.linspace(25, 65, 200)
        y_fit_total = double_gaussian(x_fit, *popt)
        y_fit1 = popt[0] * np.exp(-(x_fit - popt[1]) ** 2 / (2 * popt[2] ** 2))
        y_fit2 = popt[3] * np.exp(-(x_fit - popt[4]) ** 2 / (2 * popt[5] ** 2))

        # ax.plot(x_fit, y_fit_total, 'r-', linewidth=2, label='Total fit')
        ax.plot(x_fit, y_fit1, '--', color='black', linewidth=1.5, label='Peak 1')
        ax.plot(x_fit, y_fit2, '--', color='red', linewidth=1.5, label='Peak 2')

        params1 = format_fit_params(popt[0], popt[1], popt[2])
        params2 = format_fit_params(popt[3], popt[4], popt[5])

        # ax.text(0.02, 0.98, 'Peak 1:\n' + params1,
        #         transform=ax.transAxes,
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        #
        # ax.text(0.98, 0.98, 'Peak 2:\n' + params2,
        #         transform=ax.transAxes,
        #         horizontalalignment='right',
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        element_name = ELEMENTS.get(Z, "Unknown Element")

        ax.set_xlabel('Fragment Charge (Z)', fontsize=16)
        ax.set_ylabel('Probability Density', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f'$^{{{Z + N}}}${element_name}', fontsize=20)
        ax.set_ylim(0, 0.3)
        ax.grid(True, alpha=0.3)
        #ax.legend()

        return popt

    print(f"Warning: Could not fit double Gaussian curve to the data")
    ax.set_xlabel('Fragment Charge (Z)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Fragment Charge Distribution (Z={Z}, N={N})')
    ax.set_ylim(0, 0.3)
    ax.grid(True, alpha=0.3)
    #ax.legend(False)
    return None


def process_multiple_files(filenames, energy):
    """Process multiple endpoint files and create a combined plot of Z histograms"""
    fig, axes = plt.subplots(5, 1, figsize=(12, 20))
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
            ax.axvline(x=reference_peak, color='black', linestyle=':', linewidth=2,
                       label='Z=94 Peak 2' if ax == axes[0] else "")
            # ax.legend()

    output_filename = f'combined_z_histograms_{energy}.png'

    plt.tight_layout()
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


excitation_energies = [8.0, 20.0, 26.0]

for excitation_energy in excitation_energies:
    # Example usage with 5 files
    input_files = [
        f"98_152_{excitation_energy}_0_1000_FG_0.0_Endpoints.txt",
        f"96_150_{excitation_energy}_0_1000_FG_0.0_Endpoints.txt",
        f"94_146_{excitation_energy}_0_1000_FG_0.0_Endpoints.txt",
        f"92_144_{excitation_energy}_0_1000_FG_0.0_Endpoints.txt",
        f"90_140_{excitation_energy}_0_1000_FG_0.0_Endpoints.txt"
    ]

    process_multiple_files(input_files, excitation_energy)
