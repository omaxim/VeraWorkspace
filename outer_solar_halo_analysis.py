import pandas as pd
import argparse
from mass_density_tools import get_mass_density_stellar_halo_inputs, compute_mass_density_histogram

def main():
    parser = argparse.ArgumentParser(description="Compute mass density histograms for stellar halos.")
    parser.add_argument('--base_path', type=str, required=True, help='Path to the simulation output directory')
    parser.add_argument('--snapnum', type=int, default=99, help='Snapshot number')
    parser.add_argument('--dr', type=float, default=1.0, help='Bin width in kpc')
    parser.add_argument('--r_min', type=float, default=10.0, help='Minimum radius in kpc')
    parser.add_argument('--threshold', type=float, default=2.0, help='Threshold factor, pints over the power law this x times are not counted in the fit')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for feather file')

    args = parser.parse_args()

    # Load data
    star_positions, star_masses, tree, GroupPos, R200Crit, M200Crit = get_mass_density_stellar_halo_inputs(
        basePath=args.base_path,
        SnapNum=args.snapnum,
        PartNum=4
    )

    # Compute histograms
    reslist = []
    for i in range(GroupPos.shape[0]):
        if M200Crit[i] < 10**13:
            continue
        r_max = R200Crit[i]
        center = GroupPos[i]
        result = compute_mass_density_histogram(
            star_positions, star_masses, tree,center,
            args.r_min, r_max, args.dr,
            factor_threshold=args.threshold
        )
        result['HaloID'] = i
        reslist.append(result)

    # Save output
    df = pd.DataFrame(reslist)
    filename = f"{args.output_dir}/dr{args.dr}_rmin{args.r_min}_threshold{args.threshold}_snap{args.snapnum}.feather"
    df.to_feather(filename)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    main()

#python outer_solar_halo_analysis.py --base_path ../../../../virgotng/universe/IllustrisTNG/TNG100-2/output --snapnum 99 --dr 1 --r_min 10 --threshold 2 --output_dir TNG100-2


#base_path = "../../../../virgotng/universe/IllustrisTNG/TNG100-2/output"