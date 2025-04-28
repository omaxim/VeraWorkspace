import numpy as np
import illustris_python as il
import scipy.spatial as sc
from scipy.stats import linregress
import matplotlib.pyplot as plt
from cosmo_units import conversions

def get_mass_density_stellar_halo_inputs(basePath, SnapNum, PartNum):
    """
    Loads the necessary data for computing mass densities: star_positions, star_masses, KDTree, halo positions, and halo radii.
    
    Parameters:
        basePath (str): Path to the dataset.
        SnapNum (int): Snapshot number.
        PartNum (int): Particle number for star data.
        
    Returns:
        tuple: (star_positions, star_masses, tree, halo_positions, halo_radii)
            - star_positions: Array of star positions.
            - star_masses: Array of star masses.
            - tree: KDTree built from star positions.
            - halo_positions: Positions of centres of halos.
            - R200Crit: Virial radii of halos.
    """
    star_positions = il.snapshot.loadSubset(basePath, SnapNum, PartNum, fields="Coordinates")
    star_masses = il.snapshot.loadSubset(basePath, SnapNum, PartNum, fields="Masses") * 1e10 / 0.704
    halos = il.groupcat.loadHalos(basePath, snapNum=SnapNum)
    halo_masses = halos['Group_M_Crit200'] * 1e10 / 0.704
    halo_positions = halos['GroupPos']
    halo_radii = halos['Group_R_Crit200']
    header = il.groupcat.loadHeader(basePath, SnapNum)
    box_size = header['BoxSize']
    tree = sc.cKDTree(star_positions, boxsize=box_size)  # Create KDTree for efficient nearest-neighbor search
    return star_positions, star_masses, tree, halo_positions, halo_radii, halo_masses


def compute_mass_density_histogram(star_positions, star_masses, tree, r_min_log, r_max_log, dr, center):
    """
    Computes log10 mass density in radial bins from a center, considering periodic boundary conditions.
    
    Parameters:
        star_positions (Nx3): Star positions.
        star_masses (N,): Star masses.
        r_min_log, r_max_log (float): Log10 bounds of radius.
        dr (float): Bin width in log10 space.
        center (3,): Center of the spherical bins.
        box_size (float): Size of the periodic box (assumed cubic).
    """
    r_min = 10**r_min_log  # Convert from log10 to linear space
    r_max = 10**r_max_log
    within_rmax = tree.query_ball_point(center, r_max)
    within_rmin = tree.query_ball_point(center, r_min)
    indices = np.setdiff1d(within_rmax,within_rmin,assume_unique=True)
    if indices.size==0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        star_positions_subset = star_positions[indices]
        star_masses_subset = star_masses[indices]
        total_star_mass = star_masses_subset.sum()
        # Apply minimum-image convention to get displacement vectors
        dx = star_positions_subset - center
        distances = np.linalg.norm(dx, axis=1)

        # Bin setup
        log_r_bins = np.arange(r_min_log, r_max_log + dr, dr)
        r_bins = 10**log_r_bins
        bin_indices = np.digitize(distances, r_bins) - 1
        num_bins = len(r_bins) - 1
        mass_in_bin = np.zeros(num_bins)


        # Mass per shell
        for i in range(num_bins):
            in_bin = bin_indices == i
            mass_in_bin[i] = np.sum(star_masses_subset[in_bin])

        # Volumes and densities
        volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
        mass_density = mass_in_bin / volumes

        log_mass_densities = np.log10(mass_density)
        bin_centers = (log_r_bins[:-1] + log_r_bins[1:]) / 2

        # Linear fit
        valid = log_mass_densities > 0
        if np.any(valid):
            slope, intercept, *_ = linregress(bin_centers[valid], log_mass_densities[valid])
        else:
            slope, intercept = np.nan, np.nan

        return log_mass_densities[valid], bin_centers[valid], slope, intercept, total_star_mass

def plot_mass_density_slope_histogram(star_positions, star_masses, tree, r_min_log, r_max_log, dr, center):
    """
    Plots the log10 mass density as a function of radius and overlays the linear regression fit.

    Parameters:
        star_positions (numpy.ndarray): Array of star positions.
        star_masses (numpy.ndarray): Array of star masses.
        tree (cKDTree): KDTree of star positions for faster distance queries.
        r_min_log (float): Log10 of the minimum radius for the radial bins.
        r_max_log (float): Log10 of the maximum radius for the radial bins.
        dr (float): Log10 width of each radial bin.
        center (array-like): The center position [x, y, z] from which to compute the radial distances.
    """
    # Compute mass density
    log_masses, bins, slope, intercept, total_star_mass = compute_mass_density_histogram(star_positions, star_masses, tree, r_min_log, r_max_log, dr, center)

    # Convert log bins back to linear scale for plotting
    r_values = 10**bins
    fitted_values = 10**(slope * bins + intercept)  # Convert log fit back to linear scale

    # Create the plot
    plt.figure(figsize=(7, 5))
    plt.scatter(r_values, 10**log_masses, label="Data", color="blue", alpha=0.7)
    plt.plot(r_values, fitted_values, label=f"Fit: α = {slope:.2f}", color="red", linestyle="--")

    # Formatting
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$r$ [ckpc/h]", fontsize=12)
    plt.ylabel(r"$\rho$ [$M_{\odot}$ (ckpc/h)$^{-3}$]", fontsize=12)
    plt.title("Mass Density Profile", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Show the plot
    plt.show()