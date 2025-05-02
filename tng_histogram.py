def spherical_histogram(coords, weights=None, rmin=None, rmax=None, nbins=10, 
                        log_bins=False, density=False, 
                        poisson_error=False, cumulative=False, cumulative_desc=False,
                        min_points=1):
    """
    Compute a spherical (radial) histogram of 3D data.

    Parameters
    ----------
    coords : array-like, shape (N,3)
        Coordinates of points (e.g., simulation particle positions). Each row is (x, y, z).
    weights : array-like, shape (N,), optional
        Weights for each point (e.g., mass). If None, all weights = 1.
    rmin, rmax : float, optional
        Minimum and maximum radius for bins. Defaults to [0, max_radius].
        If log_bins=True, rmin must be > 0.
    nbins : int, optional
        Number of radial bins (default 10).
    log_bins : bool, optional
        If True, use logarithmic spacing of radii. Otherwise linear spacing.
    density : bool, optional
        If True, also compute density = count/volume (and weight_density).
    poisson_error : bool, optional
        If True, include Poisson error estimates for counts and weighted sums.
    cumulative : bool, optional
        If True, compute cumulative counts and weights in ascending order.
    cumulative_desc : bool, optional
        If True, compute cumulative counts/weights in descending order (from outer bins inward).
    min_points : int, optional
        Minimum number of points required in a bin. If a bin has fewer points, 
        the indices of those points are collected (combined) in the output.
        Default is 1 (no special handling).

    Returns
    -------
    result : dict
        Dictionary with keys:
          - 'bin_edges': array of length (nbins+1), the edges of radial bins.
          - 'bin_centers': array of length (nbins,), center radius of each bin.
          - 'volume': array of length (nbins,), volume of each spherical shell.
          - 'counts': array of length (nbins,), number of points in each bin.
          - 'sum_weights': array of length (nbins,), total weight in each bin.
          - 'mean_r': array of length (nbins,), mean radius of points in each bin.
          - 'median_r': array of length (nbins,), median radius (50th percentile).
          - 'q1_r': array of length (nbins,), 25th percentile of radii in each bin.
          - 'q3_r': array of length (nbins,), 75th percentile of radii in each bin.
          - Optional entries if enabled:
            - 'density': counts[i]/volume[i] (if density=True).
            - 'weight_density': sum_weights[i]/volume[i] (if density=True).
            - 'count_err': sqrt(counts) (if poisson_error=True).
            - 'weight_err': sqrt(sum(weights^2) in bin) (if poisson_error=True).
            - 'cumulative_counts': cumulative sum of counts (if cumulative=True).
            - 'cumulative_weights': cumulative sum of weights (if cumulative=True).
            - 'cumulative_counts_desc': descending cumulative (if cumulative_desc=True).
            - 'cumulative_weights_desc': descending cumulative (if cumulative_desc=True).
            - 'low_count_indices': combined array of point indices in bins with count < min_points.

    Notes
    -----
    - Radial bins are defined between rmin and rmax.  If log_bins is True, 
      bins are spaced logarithmically (`np.logspace`); otherwise linearly (`np.linspace`).
    - Bin volumes are given by V_i = (4/3)*π*(r_{i+1}^3 - r_i^3)&#8203;:contentReference[oaicite:0]{index=0}.
    - Weighted histograms use `np.histogram` with weights; each point's weight contributes to its bin&#8203;:contentReference[oaicite:1]{index=1}.
    - The median and quartiles are computed using NumPy percentiles of the radii in each bin.
    """
    import numpy as np

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be an (N,3) array of 3D positions")
    N = coords.shape[0]

    # Compute radial distances
    r = np.linalg.norm(coords, axis=1)
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.asarray(weights)
        if weights.shape != (N,):
            raise ValueError("weights must have the same length as coords")

    # Determine rmin and rmax if not provided
    if rmin is None:
        rmin_val = 0.0 if not log_bins else (np.min(r[r>0]) if np.any(r>0) else 1e-10)
    else:
        rmin_val = float(rmin)
    if rmax is None:
        rmax_val = r.max()
    else:
        rmax_val = float(rmax)
    if rmax_val <= rmin_val:
        raise ValueError("rmax must be larger than rmin")

    # Logarithmic bins require rmin > 0
    if log_bins:
        if rmin_val <= 0:
            raise ValueError("rmin must be > 0 for log bins")
        edges = np.logspace(np.log10(rmin_val), np.log10(rmax_val), nbins+1)
    else:
        edges = np.linspace(rmin_val, rmax_val, nbins+1)

    # Compute histogram: counts and sum of weights
    counts, _ = np.histogram(r, bins=edges)
    sum_w, _ = np.histogram(r, bins=edges, weights=weights)
    bin_edges = edges
    # Compute bin centers as midpoints
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    # Compute volume of each spherical shell: V = 4/3 π (r2^3 - r1^3)&#8203;:contentReference[oaicite:2]{index=2}
    vol = (4.0/3.0) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    # Prepare result arrays
    median_r = np.zeros(nbins)
    mean_r = np.zeros(nbins)
    q1_r = np.zeros(nbins)
    q3_r = np.zeros(nbins)

    # Bin points by index for per-bin stats; use digitize to assign bin indices (1..nbins)
    bin_idx = np.digitize(r, bin_edges) - 1
    # Fix values exactly equal to rmax fall outside; bring them to last bin
    bin_idx[bin_idx == nbins] = nbins - 1

    for i in range(nbins):
        mask = (bin_idx == i)
        if np.any(mask):
            ri = r[mask]
            # Compute statistics only on non-empty bins
            median_r[i] = np.percentile(ri, 50)
            q1_r[i] = np.percentile(ri, 25)
            q3_r[i] = np.percentile(ri, 75)
            mean_r[i] = ri.mean()
        else:
            median_r[i] = np.nan
            q1_r[i] = np.nan
            q3_r[i] = np.nan
            mean_r[i] = np.nan

    result = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'volume': vol,
        'counts': counts,
        'sum_weights': sum_w,
        'mean_r': mean_r,
        'median_r': median_r,
        'q1_r': q1_r,
        'q3_r': q3_r
    }

    # Optionally compute densities
    if density:
        # Avoid division by zero: where vol is zero, set density to nan
        dens = counts / vol
        wdens = sum_w / vol
        dens[vol == 0] = np.nan
        wdens[vol == 0] = np.nan
        result['density'] = dens
        result['weight_density'] = wdens

    # Poisson error: sqrt(N) for counts, sqrt(sum(w^2)) for weights
    if poisson_error:
        count_err = np.sqrt(result['counts'])
        # Compute sum of weights squared per bin
        sumw2 = np.histogram(r, bins=bin_edges, weights=weights**2)[0]
        weight_err = np.sqrt(sumw2)
        result['count_err'] = count_err
        result['weight_err'] = weight_err

    # Cumulative sums (ascending)
    if cumulative:
        result['cumulative_counts'] = np.cumsum(counts)
        result['cumulative_weights'] = np.cumsum(sum_w)
    # Descending cumulative (from outer bins inward)
    if cumulative_desc:
        result['cumulative_counts_desc'] = np.cumsum(counts[::-1])[::-1]
        result['cumulative_weights_desc'] = np.cumsum(sum_w[::-1])[::-1]

    # Identify low-count bins if min_points > 1
    if min_points is not None and min_points > 1:
        low_mask = counts < min_points
        if np.any(low_mask):
            # Collect indices of all points in any low-count bin
            idx_low = np.where(np.isin(bin_idx, np.where(low_mask)[0]))[0]
            result['low_count_indices'] = idx_low
        else:
            result['low_count_indices'] = np.array([], dtype=int)

    return result
