import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def gaussian(x, amplitude, mean, stddev):
    """Gaussian function for curve fitting"""
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * stddev ** 2))


def process_file(filename):
    # Extract Z and N from filename
    parts = filename.split('_')
    Z = int(parts[0])
    N = int(parts[1])
    A = Z + N

    # Read data from file
    volumes = []
    with open(filename, 'r') as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) >= 16:
                x = float(columns[15])
                # Add both fragments
                volumes.extend([x * A, (1 - x) * A])

    # Create histogram
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(volumes, bins=A, range=(0, A), density=True, alpha=0.6,
                               color='skyblue', edgecolor='black')

    # Fit Gaussian to the data
    bin_centers = (bins[:-1] + bins[1:]) / 2
    try:
        # Initial guess for parameters [amplitude, mean, standard deviation]
        p0 = [np.max(counts), np.mean(volumes), np.std(volumes)]
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)

        # Generate points for smooth curve
        x_fit = np.linspace(0, A, 200)
        y_fit = gaussian(x_fit, *popt)

        # Plot fitted curve
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Gaussian fit')
    except:
        print("Warning: Could not fit Gaussian curve to the data")

    plt.xlabel('Fragment Mass (A)')
    plt.ylabel('Probability Density')
    plt.title(f'Fragment Mass Distribution (Z={Z}, N={N}, A={A})')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save plot
    output_filename = filename.replace('Endpoints', 'Histogram')
    output_filename = os.path.splitext(output_filename)[0] + '.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()
    plt.close()


# Process the file
input_file = "90_140_20.0_0_1000_FG_0.0_Endpoints.txt"
process_file(input_file)