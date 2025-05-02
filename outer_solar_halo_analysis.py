import pandas as pd
import argparse
import os
from mass_density_tools import get_mass_density_stellar_halo_inputs, compute_mass_density_histogram

def main():
    parser = argparse.ArgumentParser(description="Compute mass density histograms for stellar halos.")
    parser.add_argument('--base_path', type=str, required=True, help='Path to the simulation output directory')
    parser.add_argument('--sim_name', type=str, required=True, help='Name of the simulation (used for output subdir)')
    parser.add_argument('--snapnum', type=int, default=99, help='Snapshot number')
    parser.add_argument('--dr', type=float, default=1.0, help='Bin width in kpc')
    parser.add_argument('--r_min', type=float, default=10.0, help='Minimum radius in kpc')
    parser.add_argument('--threshold', type=float, default=2.0, help='Threshold factor for density fitting')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.join("outputs", args.sim_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    star_positions, star_masses, tree, GroupPos, R200Crit, M200Crit = get_mass_density_stellar_halo_inputs(
        basePath=args.base_path,
        SnapNum=args.snapnum,
        PartNum=4
    )

    # Compute histograms
    reslist = []
    for i in range(GroupPos.shape[0]):
        if M200Crit[i] < 1e13:
            continue
        r_max = R200Crit[i]
        center = GroupPos[i]
        result = compute_mass_density_histogram(
            star_positions, star_masses, tree, center,
            args.r_min, r_max, args.dr,
            factor_threshold=args.threshold
        )
        result['HaloID'] = i
        reslist.append(result)

    # Save output
    filename = f"{output_dir}/dr{args.dr}_rmin{args.r_min}_threshold{args.threshold}_snap{args.snapnum}.feather"
    df = pd.DataFrame(reslist)
    df.to_feather(filename)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
