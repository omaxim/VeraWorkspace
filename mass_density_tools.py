import numpy as np
import illustris_python as il
import scipy.spatial as sc
from scipy.stats import linregress
from tng_histogram import spherical_histogram
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


def compute_mass_density_histogram(
    star_positions, star_masses, tree, center,
    r_min, r_max, dr, factor_threshold,
    **kwargs
):
    """
    Computes mass density in radial shells using log bins and fits a slope in log-log space.
    Utilizes `spherical_histogram` for flexible histogramming.

    Parameters
    ----------
    star_positions : (N, 3) array
        3D positions of stars.
    star_masses : (N,) array
        Masses of stars.
    center : (3,) array
        Center position for the radial histogram.
    r_min_log, r_max_log : float
        Log10 of min and max radius for binning.
    dr : float
        Log10 bin width.
    **kwargs :
        Additional keyword arguments passed to `spherical_histogram`.

    Returns
    -------
    result : dict
        Dictionary with keys:
          - 'log_r': bin center radii in log10
          - 'log_density': log10 of mass density
          - 'slope': slope of log-log fit
          - 'intercept': intercept of log-log fit
          - 'total_mass': total stellar mass considered
          - All other keys from `spherical_histogram`
    """

    nbins = int(np.ceil((r_max - r_min) / dr))

    shifted_positions = star_positions - center
    within_rmax = tree.query_ball_point(center, r_max)
    within_rmin = tree.query_ball_point(center, r_min)
    indices = np.setdiff1d(within_rmax,within_rmin,assume_unique=True)

    result = spherical_histogram(
        coords=shifted_positions[indices],
        weights=star_masses[indices],
        rmin=r_min,
        rmax=r_max,
        nbins=nbins,
        density=True,
        **kwargs
    )
    weight_density = result.get("weight_density")
    log_r = np.log10(result["median_r"])
    with np.errstate(divide='ignore', invalid='ignore'):
        log_density = np.log10(weight_density)
    
    valid = np.isfinite(log_density)
    if np.count_nonzero(valid) >= 2:
        slope, intercept, *_ = linregress(log_r[valid], log_density[valid])
    else:
        slope, intercept = np.nan, np.nan

    residuals = log_density - (slope * log_r + intercept)
    # Threshold for identifying satellite bumps
    residual_threshold = np.log10(factor_threshold)
    satellite_mask = residuals < residual_threshold

    # Ensure enough data for final fit
    if np.sum(satellite_mask) >= 2:
        slope, intercept, _, _, _ = linregress(log_r[satellite_mask], log_density[satellite_mask])
    else:
        print("Warning: Not enough points after masking, using initial fit.")



    return {
        'log_r': log_r,
        'log_density': log_density,
        'slope': slope,
        'intercept': intercept,
        'satellite_mask': satellite_mask,
        **result
    }

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