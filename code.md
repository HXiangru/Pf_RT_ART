# Inferring Replication Timing and Identifying Artemisinin Damage in Plasmodium falciparum
#### Author: XiangRu Huang


---

## rt_01_rfd

```python
#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from typing import Dict
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

# Simplified directory structure for public sharing
# Assumes the script is run from a base directory containing 'data/', 'results/', and 'plots/'
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
PLOTS_DIR = BASE_DIR / 'plots'

# Create output directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input BED files (place your data in the 'data' directory)
LEFT_FORKS_PATH = DATA_DIR / 'left_forks.bed'
RIGHT_FORKS_PATH = DATA_DIR / 'right_forks.bed'

# Bin sizes for analysis
BIN_SIZES = [5000, 10000, 20000]  # in base pairs
RESOLUTION_LABELS = {5000: '5kb', 10000: '10kb', 20000: '20kb'}

# Plotting colors
COLOR_RIGHT_DOMINANT = '#f7de98'  # Softer yellow for RFD < 0.5
COLOR_LEFT_DOMINANT = '#add5a2'   # Softer green for RFD >= 0.5
COLOR_AUTOCORR = {
    5000: '#41b349',   # Green
    10000: '#2775b6',  # Blue
    20000: '#fcc307'   # Orange
}

# =========================== DATA LOADING ===========================

def read_bed_file(path: Path) -> pd.DataFrame:
    """Reads a simple 3-column BED file (chrom, start, end)."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        comment='#',
        usecols=[0, 1, 2],
        names=['chrom', 'start', 'end'],
        dtype={'chrom': str}
    )
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna().astype({'start': int, 'end': int}).reset_index(drop=True)
    return df

# =========================== RFD COMPUTATION ===========================

def compute_rfd(fork_data: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """
    Computes Replication Fork Directionality (RFD) in fixed genomic bins.
    RFD = (left_forks + 1) / (left_forks + right_forks + 2)
    """
    all_binned_data = []
    chrom_lengths = fork_data.groupby('chrom')['end'].max()

    for chrom, length in chrom_lengths.items():
        bin_edges = np.arange(0, int(length) + bin_size, bin_size)
        num_bins = len(bin_edges) - 1
        
        chrom_forks = fork_data[fork_data['chrom'] == chrom]
        
        left_counts = np.zeros(num_bins, dtype=int)
        right_counts = np.zeros(num_bins, dtype=int)
        
        if not chrom_forks.empty:
            starts = chrom_forks['start'].values
            types = chrom_forks['fork_type'].values
            
            # Assign each fork's start coordinate to a bin index
            bin_indices = np.searchsorted(bin_edges, starts, side='right') - 1
            
            # Ensure indices are within the valid range [0, num_bins-1]
            valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
            valid_indices = bin_indices[valid_mask]
            valid_types = types[valid_mask]

            if valid_indices.size > 0:
                np.add.at(left_counts, valid_indices[valid_types == 'left'], 1)
                np.add.at(right_counts, valid_indices[valid_types == 'right'], 1)
        
        rfd_values = (left_counts + 1) / (left_counts + right_counts + 2)
        
        binned_df = pd.DataFrame({
            'chrom': chrom,
            'start': bin_edges[:-1],
            'end': bin_edges[1:],
            'mid': (bin_edges[:-1] + bin_edges[1:]) // 2,
            'left_counts': left_counts,
            'right_counts': right_counts,
            'RFD': rfd_values
        })
        all_binned_data.append(binned_df)

    if not all_binned_data:
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'mid', 'left_counts', 'right_counts', 'RFD'])
    
    return pd.concat(all_binned_data, ignore_index=True)

# =========================== PLOTTING FUNCTIONS ===========================

def plot_rfd_by_resolution(rfd_data: pd.DataFrame, resolution_label: str, output_path: Path):
    """Creates a multi-page PDF with one chromosome per page."""
    chromosomes = sorted(rfd_data['chrom'].unique(), key=lambda c: int(c) if c.isdigit() else c)
    
    with PdfPages(output_path) as pdf:
        for chrom in chromosomes:
            chrom_df = rfd_data[rfd_data['chrom'] == chrom].sort_values('mid')
            if chrom_df.empty or (chrom_df['left_counts'].sum() + chrom_df['right_counts'].sum() == 0):
                continue

            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.suptitle(f"Chromosome {chrom} - RFD ({resolution_label})", fontsize=10)
            
            # Filter to only plot bins containing forks
            data_mask = (chrom_df['left_counts'] + chrom_df['right_counts']) > 0
            plot_df = chrom_df[data_mask]

            x_mb = plot_df['start'].values / 1e6
            widths_mb = (plot_df['end'] - plot_df['start']).values / 1e6
            rfd_vals = plot_df['RFD'].values
            
            heights = np.abs(rfd_vals - 0.5)
            bottoms = np.minimum(rfd_vals, 0.5)
            colors = [COLOR_LEFT_DOMINANT if v >= 0.5 else COLOR_RIGHT_DOMINANT for v in rfd_vals]
            
            ax.bar(x_mb, heights, width=widths_mb, bottom=bottoms, color=colors, align='edge')
            
            ax.set_ylabel('RFD')
            ax.set_ylim(0, 1)
            ax.axhline(0.5, linestyle='--', color='black', linewidth=0.6)
            ax.set_xlabel('Genomic Position (Mb)')
            
            legend_elements = [
                Patch(facecolor=COLOR_RIGHT_DOMINANT, label='Right-moving Fork Dominant'),
                Patch(facecolor=COLOR_LEFT_DOMINANT, label='Left-moving Fork Dominant')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved PDF: {output_path}")

def plot_rfd_by_chromosome(chrom: str, rfd_data_by_res: Dict[int, pd.DataFrame], output_path: Path):
    """Creates a single PDF page for one chromosome, stacking all resolutions."""
    fig, axes = plt.subplots(len(BIN_SIZES), 1, figsize=(12, 10), sharex=True)
    if len(BIN_SIZES) == 1: axes = [axes] # Ensure axes is iterable for single resolution
    fig.suptitle(f"Chromosome {chrom} - RFD Comparison", fontsize=20)

    for ax, bin_size in zip(axes, BIN_SIZES):
        df = rfd_data_by_res[bin_size]
        chrom_df = df[df['chrom'] == chrom].sort_values('mid')
        
        ax.set_title(f"{RESOLUTION_LABELS[bin_size]} bins", loc='center', fontsize=16)
        ax.set_ylabel('RFD')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, linestyle='--', color='black', linewidth=0.6)

        if chrom_df.empty or (chrom_df['left_counts'].sum() + chrom_df['right_counts'].sum() == 0):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
            
        data_mask = (chrom_df['left_counts'] + chrom_df['right_counts']) > 0
        plot_df = chrom_df[data_mask]

        x_mb = plot_df['start'].values / 1e6
        widths_mb = (plot_df['end'] - plot_df['start']).values / 1e6
        rfd_vals = plot_df['RFD'].values
        
        heights = np.abs(rfd_vals - 0.5)
        bottoms = np.minimum(rfd_vals, 0.5)
        colors = [COLOR_LEFT_DOMINANT if v >= 0.5 else COLOR_RIGHT_DOMINANT for v in rfd_vals]
        
        ax.bar(x_mb, heights, width=widths_mb, bottom=bottoms, color=colors, align='edge')
        
        legend_elements = [
            Patch(facecolor=COLOR_RIGHT_DOMINANT, label='Right Fork Dom.'),
            Patch(facecolor=COLOR_LEFT_DOMINANT, label='Left Fork Dom.')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Genomic Position (Mb)')
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for suptitle
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chromosome plot: {output_path}")

def calculate_autocorrelation(rfd_df: pd.DataFrame, max_lag_bins: int = 50) -> np.ndarray:
    """Calculates and averages autocorrelation across all chromosomes in a dataframe."""
    all_autocorrs = []
    
    for chrom in rfd_df['chrom'].unique():
        rfd_series = rfd_df.loc[rfd_df['chrom'] == chrom, 'RFD'].dropna()
        
        if len(rfd_series) < max_lag_bins:
            continue
            
        autocorr_chrom = [rfd_series.autocorr(lag=i) for i in range(max_lag_bins)]
        all_autocorrs.append(autocorr_chrom)

    if not all_autocorrs:
        return np.full(max_lag_bins, np.nan)
        
    return np.nanmean(np.array(all_autocorrs, dtype=float), axis=0)

def plot_autocorrelation_combined(rfd_data_by_res: Dict[int, pd.DataFrame], output_path: Path):
    """Generates a combined plot with three panels for autocorrelation at different resolutions."""
    fig, axes = plt.subplots(len(BIN_SIZES), 1, figsize=(8, 12), sharex=True)
    if len(BIN_SIZES) == 1: axes = [axes]
    fig.suptitle('RFD Autocorrelation at Different Resolutions', fontsize=14)
    
    for i, (bin_size, rfd_df) in enumerate(rfd_data_by_res.items()):
        ax = axes[i]
        mean_autocorrs = calculate_autocorrelation(rfd_df)
        
        lag_distance_kb = bin_size / 1000
        lags_kb = np.arange(len(mean_autocorrs)) * lag_distance_kb
        
        ax.plot(lags_kb, mean_autocorrs, marker='o', linestyle='-', 
                color=COLOR_AUTOCORR[bin_size], markersize=4, linewidth=1.5)
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
        ax.set_title(f'{RESOLUTION_LABELS[bin_size]} bins', fontsize=11)
        ax.set_ylabel('Autocorrelation Coefficient')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-0.3, 0.3) # Unify y-axis for comparison

    axes[-1].set_xlabel('Lag Distance (kb)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined autocorrelation plot: {output_path}")

# =========================== MAIN EXECUTION ===========================

def main():
    """Main workflow to load data, compute RFD, and generate all plots."""
    print("Starting RFD analysis workflow...")
    
    # Load and combine fork data
    try:
        left_forks = read_bed_file(LEFT_FORKS_PATH)
        left_forks['fork_type'] = 'left'
        right_forks = read_bed_file(RIGHT_FORKS_PATH)
        right_forks['fork_type'] = 'right'
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your input files ('left_forks.bed', 'right_forks.bed') are in the 'data/' directory.")
        return

    all_forks = pd.concat([left_forks, right_forks], ignore_index=True)
    print(f"Loaded {len(left_forks)} left forks and {len(right_forks)} right forks.")
    
    chromosomes = sorted(all_forks['chrom'].unique(), key=lambda c: int(c) if c.isdigit() else c)
    print(f"Found chromosomes: {', '.join(chromosomes)}")

    rfd_results_by_resolution = {}

    # Process each resolution
    for b_size in BIN_SIZES:
        res_label = RESOLUTION_LABELS[b_size]
        print(f"\nProcessing resolution: {res_label}")
        
        rfd_df = compute_rfd(all_forks, b_size)
        rfd_results_by_resolution[b_size] = rfd_df
        
        # Save RFD table
        output_table_path = RESULTS_DIR / f"rfd_{res_label}.tsv"
        rfd_df[['chrom', 'start', 'end', 'RFD']].to_csv(output_table_path, sep='\t', index=False)
        print(f"Saved RFD table: {output_table_path}")

        # Save per-resolution, multi-page RFD plot
        output_pdf_path = PLOTS_DIR / f"rfd_by_chrom_{res_label}.pdf"
        plot_rfd_by_resolution(rfd_df, res_label, output_pdf_path)

    # Save one PDF per chromosome, stacking all resolutions
    print("\nGenerating multi-resolution plots for each chromosome...")
    for chrom in chromosomes:
        chrom_pdf_path = PLOTS_DIR / f"chr{chrom}_rfd_comparison.pdf"
        plot_rfd_by_chromosome(chrom, rfd_results_by_resolution, chrom_pdf_path)

    # Generate autocorrelation plots
    print("\nGenerating RFD autocorrelation plots...")
    combined_autocorr_path = PLOTS_DIR / "rfd_autocorrelation_combined.pdf"
    plot_autocorrelation_combined(rfd_results_by_resolution, combined_autocorr_path)
    
    print("\nWorkflow complete. Outputs are in:")
    print(f"  Tables: {RESULTS_DIR}")
    print(f"  Plots:  {PLOTS_DIR}")

if __name__ == '__main__':
    main()
```







---

## rt_02_rt_rfd

```python

import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List
import matplotlib as mpl

# Ensure fonts are embeddable in PDFs
mpl.rcParams['pdf.fonttype'] = 42

# --- 1. Configuration & Global Constants ---

# Simplified relative paths for portability
BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"
PLOT_DIR = BASE_DIR / "plots"

# Analysis parameters
RESOLUTIONS: List[int] = [5, 10, 20]
OUTPUT_PREFIX: str = 'rt_'
CHROMOSOMES: List[str] = [str(i) for i in range(1, 15)]

# Column name constants for clarity
COL_CHR = "chr"
COL_START = "start"
COL_END = "end"
COL_RFD = "RFD"
COL_RT = "RT"

# Custom color palette for plotting
COLOR_PALETTE: Dict[int, str] = {
    5: '#41b349',   # Green
    10: '#2775b6',  # Blue
    20: '#fcc307'   # Yellow
}
COLOR_RFD: str = '#f7de98'  # Light yellow/gold for RFD background


def ensure_directories():
    """Create output directories if they don't exist."""
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Results and plots will be saved in '{RESULTS_DIR.resolve()}' and '{PLOT_DIR.resolve()}'.")


# --- 2. Core Logic: Replication Timing Calculation ---

def compute_replication_timing(rfd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Replication Timing (RT) from Replication Fork Directionality (RFD)
    data using genome-wide normalization.
    """
    print("Computing replication timing (RT) with global normalization...")
    all_chr_rt = []
    
    # Sort chromosomes numerically (e.g., '2' before '10')
    unique_chroms = sorted(rfd_df[COL_CHR].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    for chrom in unique_chroms:
        chr_data = rfd_df[rfd_df[COL_CHR] == chrom].sort_values(COL_START).reset_index(drop=True)
        if len(chr_data) < 2:
            continue
        
        # Calculate the cumulative integral of (RFD - 0.5)
        x_coords = (chr_data[COL_START] + chr_data[COL_END]).values / 2.0
        rfd_imbalance = chr_data[COL_RFD].values - 0.5
        integral = cumulative_trapezoid(rfd_imbalance, x_coords, initial=0.0)
        
        chr_data = chr_data.copy()
        chr_data["integral"] = integral
        all_chr_rt.append(chr_data)

    if not all_chr_rt:
        return pd.DataFrame()

    # Perform global normalization across all chromosomes
    df_all = pd.concat(all_chr_rt, ignore_index=True)
    global_min = df_all["integral"].min()
    global_max = df_all["integral"].max()
    global_range = global_max - global_min

    if np.isfinite(global_range) and global_range > 0:
        df_all[COL_RT] = (df_all["integral"] - global_min) / global_range
        print("Global RT scaled to [0, 1].")
    else:
        df_all[COL_RT] = 0.5
        print("Warning: Global integral range is zero or non-finite; assigned RT=0.5.")

    return df_all.drop(columns=["integral"])


# --- 3. Data Processing Workflow ---

def process_all_resolutions() -> Dict[int, pd.DataFrame]:
    """Load RFD files for each resolution, calculate RT, and return a dictionary of results."""
    print("\nStarting RT analysis...")
    all_rt_data: Dict[int, pd.DataFrame] = {}
    
    for res in RESOLUTIONS:
        print(f"--- Processing {res}kb resolution ---")
        input_file = RESULTS_DIR / f"rt_rfd_{res}kb.txt"
        output_file = RESULTS_DIR / f"rt_curve_de_{res}kb.txt"

        if not input_file.exists():
            print(f"Warning: Input file not found, skipping: {input_file.name}")
            continue
        
        try:
            rfd_data = pd.read_csv(input_file, sep=r'\s+')
            
            # Standardize column names
            rfd_col = next((c for c in rfd_data.columns if 'RFD' in c), None)
            if rfd_col is None:
                print(f"Warning: RFD column not found in {input_file.name}. Skipping.")
                continue
            
            rfd_data = rfd_data.rename(columns={rfd_col: COL_RFD, 'chrom': COL_CHR})
            rfd_data[COL_CHR] = rfd_data[COL_CHR].astype(str)
            
            # Calculate RT
            rt_data = compute_replication_timing(rfd_data)
            
            if not rt_data.empty:
                rt_data.to_csv(output_file, sep='\t', index=False)
                all_rt_data[res] = rt_data
                
        except Exception as e:
            print(f"Error processing {res}kb resolution: {e}")
            
    print(f"\nSuccessfully processed {len(all_rt_data)} resolutions.")
    return all_rt_data


# --- 4. Plotting Functions ---

def plot_chromosome_rt_curves(all_rt_data: Dict[int, pd.DataFrame], output_pdf: Path):
    """Generate a multi-page PDF showing RT curves for each chromosome at different resolutions."""
    print(f"Generating multi-resolution chromosome profiles: {output_pdf.name}")
    with PdfPages(output_pdf) as pdf:
        for chrom in CHROMOSOMES:
            fig, ax = plt.subplots(figsize=(15, 7))
            
            for res, rt_data in sorted(all_rt_data.items()):
                chr_data = rt_data[rt_data[COL_CHR] == chrom]
                if chr_data.empty:
                    continue
                
                pos_mb = chr_data[COL_START].values / 1e6
                ax.plot(pos_mb, chr_data[COL_RT], color=COLOR_PALETTE.get(res),
                        linewidth=2, label=f'{res}kb')
                ax.fill_between(pos_mb, chr_data[COL_RT], color=COLOR_PALETTE.get(res), alpha=0.15)

            ax.set_title(f'Chromosome {chrom} Replication Timing', fontsize=24)
            ax.set_xlabel('Genomic Position (Mb)', fontsize=20)
            ax.set_ylabel('Replication Timing (RT)', fontsize=20)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=18)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def plot_genome_wide_kde(all_rt_data: Dict[int, pd.DataFrame], output_image: Path):
    """Generate a kernel density estimate (KDE) plot of genome-wide RT values."""
    print(f"Generating genome-wide RT density plot: {output_image.name}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for res, rt_data in sorted(all_rt_data.items()):
        if not rt_data.empty:
            sns.kdeplot(data=rt_data, x=COL_RT, ax=ax, fill=True, alpha=0.3,
                        linewidth=2, color=COLOR_PALETTE.get(res), label=f'{res}kb')

    ax.set_title('Genome-wide Replication Time Density', fontsize=24)
    ax.set_xlabel('Replication Timing (RT)', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    ax.legend(fontsize=18)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_10kb_rt_rfd_curves(all_rt_data: Dict[int, pd.DataFrame], output_pdf: Path):
    """Generate a PDF showing both RT (line) and RFD (bars) for the 10kb resolution data."""
    print(f"Generating 10kb RT and RFD profiles: {output_pdf.name}")
    if 10 not in all_rt_data or all_rt_data[10].empty:
        print("Warning: 10kb data not found or empty. Skipping 10kb RT/RFD plot.")
        return

    data_10kb = all_rt_data[10]
    color_rt_10kb = COLOR_PALETTE.get(10)

    with PdfPages(output_pdf) as pdf:
        for chrom in CHROMOSOMES:
            fig, ax1 = plt.subplots(figsize=(15, 7))
            plt.style.use('seaborn-v0_8-whitegrid')
            
            chr_data = data_10kb[data_10kb[COL_CHR] == chrom]
            if chr_data.empty:
                ax1.text(0.5, 0.5, 'No data available for this chromosome',
                         ha='center', va='center', transform=ax1.transAxes)
            else:
                pos_mb = chr_data[COL_START].values / 1e6
                
                # Plot RFD as bars on the primary (left) y-axis
                ax1.bar(pos_mb, height=np.abs(chr_data[COL_RFD] - 0.5),
                        width=(chr_data[COL_END] - chr_data[COL_START]) / 1e6,
                        bottom=np.minimum(chr_data[COL_RFD], 0.5),
                        color=COLOR_RFD, align='edge', label='RFD (10kb)')
                ax1.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
                ax1.set_ylabel('RFD', fontsize=12)
                ax1.set_ylim(0, 1.05)
                
                # Plot RT as a line on the secondary (right) y-axis
                ax2 = ax1.twinx()
                ax2.plot(pos_mb, chr_data[COL_RT], color=color_rt_10kb,
                         linewidth=2.5, label='RT (10kb)')
                ax2.fill_between(pos_mb, chr_data[COL_RT], color=color_rt_10kb, alpha=0.15)
                ax2.set_ylabel('Replication Timing (RT)', fontsize=12)
                ax2.set_ylim(-0.05, 1.05)
                ax2.grid(False)

                # Combine legends
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax2.legend(h1 + h2, l1 + l2, loc='upper right')

            ax1.set_xlabel('Genomic Position (Mb)', fontsize=12)
            fig.suptitle(f'Chromosome {chrom} - 10kb Resolution', fontsize=16)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# --- 5. Main Execution Block ---

def main():
    """Main function to run the complete analysis and plotting pipeline."""
    print("="*60)
    print("Replication Timing (RT) Analysis and Plotting Script")
    print("="*60)

    ensure_directories()
    all_rt_data = process_all_resolutions()

    if not all_rt_data:
        print("\nNo data processed. Exiting.")
        return

    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    # Generate plots
    plot_chromosome_rt_curves(all_rt_data, PLOT_DIR / f"{OUTPUT_PREFIX}chromosome_profiles_multires.pdf")
    plot_genome_wide_kde(all_rt_data, PLOT_DIR / f"{OUTPUT_PREFIX}genome_wide_kde.pdf")
    plot_10kb_rt_rfd_curves(all_rt_data, PLOT_DIR / f"{OUTPUT_PREFIX}rt_rfd_profile_10kb.pdf")

    print("\n" + "="*60)
    print("Analysis complete")
    print(f"All outputs are saved in the '{RESULTS_DIR}' and '{PLOT_DIR}' directories.")
    print("="*60)

if __name__ == "__main__":
    main()



```







---

## rt_03_rt_rfd_intuitive

```python

#!/usr/-bin/env python3

from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ------------------ User Configuration ------------------
# Simplified paths for public code sharing.
# Place your input files in the DATA_DIR.
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
PLOT_DIR = Path("./plots")

# Input files should be placed in the DATA_DIR
ORIGINS_FILE = DATA_DIR / "origins.bed"
TERMINATIONS_FILE = DATA_DIR / "terminations.bed"

RESOLUTIONS: List[int] = [5, 10, 20]  # in kb
OUTPUT_PREFIX = "rt_"
CHROMOSOMES: List[str] = [f"{i:02d}" for i in range(1, 15)]

# Model parameters
BASE_SPEED = 1.0
ALPHA = 0.0
BETA = 0.0

# Plotting colors
COLOR_PALETTE: Dict[int, str] = {5: '#41b349', 10: '#2775b6', 20: '#fcc307'}
COLOR_RFD = '#f7de98'
RT_COLOR = "#2c7bb6"

# ------------------ Utilities ------------------

def ensure_directories():
    """Create output directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories ensured: {RESULTS_DIR}, {PLOT_DIR}")

def sort_key(x):
    """Sort key that handles both numeric and string values."""
    try:
        return int(str(x))
    except (ValueError, TypeError):
        return str(x)

def normalize_chr_series(s: pd.Series) -> pd.Series:
    """Extract numeric part from chromosome strings and zero-pad to 2 digits."""
    return s.astype(str).str.extract(r'(\d+)', expand=False).str.zfill(2)

# ------------------ I/O Helpers ------------------

def parse_genomic_site_bed(path: Path) -> pd.DataFrame:
    """Load a BED-like file into a DataFrame with 'chr' and 'pos' columns."""
    if not path.exists():
        return pd.DataFrame(columns=["chr", "pos"])
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None, comment='#', usecols=[0, 1, 2],
                         names=["chr", "start", "end"], dtype={"chr": str})
    except Exception as e:
        print(f"[WARN] Could not read {path.name}: {e}")
        return pd.DataFrame(columns=["chr", "pos"])
    
    df["pos"] = (pd.to_numeric(df["start"], errors="coerce").fillna(0) +
                 pd.to_numeric(df["end"], errors="coerce").fillna(0)) // 2
    df["chr"] = normalize_chr_series(df["chr"])
    return df[["chr", "pos"]].astype({"pos": int})

def generate_potential_events_from_rfd(rfd_df: pd.DataFrame, window_size: int = 5, delta_thresh: float = 0.1):
    """Infer potential origins/terminations from local changes in RFD."""
    events = []
    chroms = sorted(rfd_df["chr"].unique(), key=sort_key)
    for chrom in chroms:
        chr_rfd = rfd_df[rfd_df["chr"] == chrom].sort_values("start").reset_index(drop=True)
        if len(chr_rfd) < 3:
            continue
        
        chr_rfd["mid"] = (chr_rfd["start"] + chr_rfd["end"]) // 2
        chr_rfd["delta_rfd"] = chr_rfd["RFD"].diff().fillna(0.0)
        half_window = window_size // 2
        
        for i in range(half_window, len(chr_rfd) - half_window):
            center_delta = chr_rfd.at[i, "delta_rfd"]
            if abs(center_delta) > delta_thresh:
                event_type = "inferred_origin" if center_delta > 0 else "inferred_termination"
                events.append({
                    "chr": chrom,
                    "position": int(chr_rfd.at[i, "mid"]),
                    "event_type": event_type,
                    "delta_rfd": float(center_delta)
                })

    return pd.DataFrame(events) if events else pd.DataFrame(columns=["chr", "position", "event_type", "delta_rfd"])

# ------------------ Core Algorithm ------------------

def compute_intuitive_rt_per_resolution(rfd_df: pd.DataFrame, bin_kb: int,
                                          origins_df: pd.DataFrame = None,
                                          terminations_df: pd.DataFrame = None,
                                          base_speed: float = BASE_SPEED,
                                          alpha: float = ALPHA,
                                          beta: float = BETA) -> pd.DataFrame:
    """
    Compute RT per chromosome using an 'intuitive' algorithm with per-chromosome normalization.
    Returns a DataFrame with columns ['chr', 'start', 'end', 'RFD', 'efficiency', 'RT'].
    """
    rfd = rfd_df.copy()
    rfd.rename(columns={'chrom': 'chr'}, inplace=True)
    rfd_col = next((c for c in rfd.columns if 'rfd' in c.lower()), None)
    if rfd_col is None:
        raise KeyError("No RFD column found in the input DataFrame.")
    rfd.rename(columns={rfd_col: 'RFD'}, inplace=True)

    rfd['chr'] = normalize_chr_series(rfd['chr'])
    for col in ['start', 'end', 'RFD']:
        rfd[col] = pd.to_numeric(rfd[col], errors='coerce')

    inferred_events = generate_potential_events_from_rfd(rfd)
    results = []
    
    for chrom in sorted(rfd['chr'].unique(), key=sort_key):
        chr_rfd = rfd[rfd['chr'] == chrom].sort_values('start').dropna().reset_index(drop=True)
        if len(chr_rfd) < 2:
            continue

        chr_rfd['mid'] = ((chr_rfd['start'] + chr_rfd['end']) / 2).astype(int)
        
        # Use provided origins, otherwise fall back to inferred origins
        origin_positions = np.array([])
        if origins_df is not None and not origins_df.empty:
            op = origins_df.loc[origins_df['chr'] == chrom, 'pos'].unique()
            if len(op) >= 2:
                origin_positions = np.sort(op)
        
        if origin_positions.size < 2 and not inferred_events.empty:
            op2 = inferred_events.loc[(inferred_events['chr'] == chrom) & (inferred_events['event_type'] == 'inferred_origin'), 'position']
            if not op2.empty:
                origin_positions = np.sort(op2.unique())

        if origin_positions.size < 2:
            print(f"[INFO] Chrom {chrom}: fewer than 2 origins found, skipping.")
            continue
        
        chr_rfd['efficiency'] = chr_rfd['RFD'].diff().fillna(0.0) / float(bin_kb)
        chr_rfd['speed'] = base_speed * (1.0 + alpha * np.abs(chr_rfd['efficiency'])) * (1.0 + beta * np.abs(chr_rfd['RFD'] - 0.5))
        speed_interp = interp1d(chr_rfd['mid'], chr_rfd['speed'], kind='linear', bounds_error=False, fill_value=(chr_rfd['speed'].iloc[0], chr_rfd['speed'].iloc[-1]))

        activation_times = [0.5]
        mid_min, mid_max = chr_rfd['mid'].min(), chr_rfd['mid'].max()
        
        for i in range(len(origin_positions) - 1):
            start, end = origin_positions[i], origin_positions[i+1]
            interval_mask = (chr_rfd['mid'] >= start) & (chr_rfd['mid'] <= end)
            if not interval_mask.any():
                activation_times.append(activation_times[-1])
                continue

            interval_rfd = chr_rfd.loc[interval_mask]
            right_prop = (interval_rfd['RFD'] < 0.5).sum() / len(interval_rfd) if len(interval_rfd) > 0 else 0.5
            
            center_pos = (start + end) / 2.0
            avg_speed = float(speed_interp(center_pos))
            if not np.isfinite(avg_speed) or avg_speed <= 0: avg_speed = base_speed
            
            delta_t = (0.5 - right_prop) * (end - start) / 1000.0 / avg_speed
            activation_times.append(activation_times[-1] + delta_t)
            
        activation_times = np.array(activation_times, dtype=float)
        if np.ptp(activation_times) > 0:
            activation_times = (activation_times - activation_times.min()) / np.ptp(activation_times)
        else:
            activation_times.fill(0.5)

        rt_interp = interp1d(origin_positions, activation_times, kind='linear', bounds_error=False, fill_value=(activation_times[0], activation_times[-1]))
        chr_rfd['RT'] = rt_interp(chr_rfd['mid'].values)
        chr_rfd['RT'] = chr_rfd['RT'].clip(0.0, 1.0)
        results.append(chr_rfd[['chr', 'start', 'end', 'RFD', 'efficiency', 'RT']])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

# ------------------ Main Processing Loop ------------------

def process_all_resolutions(resolutions: List[int]) -> Dict[int, pd.DataFrame]:
    """Process all specified resolutions to compute RT profiles."""
    results_dict: Dict[int, pd.DataFrame] = {}
    origins_df = parse_genomic_site_bed(ORIGINS_FILE)
    term_df = parse_genomic_site_bed(TERMINATIONS_FILE)

    for res in resolutions:
        input_rfd = RESULTS_DIR / f"rt_rfd_{res}kb.txt"
        out_rt = RESULTS_DIR / f"rt_curve_intuitive_{res}kb.txt"
        print(f"\n--- Processing {res}kb resolution ---")
        
        if not input_rfd.exists():
            print(f"[WARN] Input file not found: {input_rfd}. Skipping.")
            continue
            
        try:
            rfd_df = pd.read_csv(input_rfd, sep=r'\s+')
            rt_df = compute_intuitive_rt_per_resolution(rfd_df, bin_kb=res, origins_df=origins_df, terminations_df=term_df)
            
            if rt_df.empty:
                print(f"[INFO] No RT values computed for {res}kb resolution.")
                continue
                
            rt_df.to_csv(out_rt, sep='\t', index=False)
            print(f"[OK] Wrote intuitive RT to {out_rt}")
            results_dict[res] = rt_df
        except Exception as e:
            print(f"[ERROR] Failed processing {input_rfd.name}: {e}")
            
    return results_dict

# ------------------ Plotting Functions ------------------

def plot_chromosome_rt_curves_multires(all_rt_data: Dict[int, pd.DataFrame], output_pdf: Path):
    """Plot multi-resolution RT profiles for each chromosome into a single PDF."""
    print(f"Generating multi-resolution chromosome profiles PDF -> {output_pdf}")
    with PdfPages(output_pdf) as pdf:
        for chrom in CHROMOSOMES:
            fig, ax = plt.subplots(figsize=(15, 7))
            found_data = False
            for res, rt_df in sorted(all_rt_data.items()):
                chr_data = rt_df[rt_df['chr'] == chrom]
                if not chr_data.empty:
                    found_data = True
                    x = chr_data['start'].values / 1e6
                    y = chr_data['RT'].values
                    ax.plot(x, y, color=COLOR_PALETTE.get(res, '#333333'), linewidth=2, label=f"{res}kb")
                    ax.fill_between(x, y, color=COLOR_PALETTE.get(res, '#333333'), alpha=0.12)
            
            if not found_data:
                plt.close(fig)
                continue
                
            ax.set_title(f"Chromosome {chrom} — Replication Timing (Intuitive Model)", fontsize=14)
            ax.set_xlabel("Genomic Position (Mb)")
            ax.set_ylabel("RT (0=Late, 1=Early)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"[OK] Saved {output_pdf}")

def plot_genome_wide_kde(all_rt_data: Dict[int, pd.DataFrame], output_pdf: Path):
    """Plot a genome-wide Kernel Density Estimate of RT values."""
    print(f"Generating genome-wide KDE plot -> {output_pdf}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plotted = any(not df.empty for df in all_rt_data.values())
    if not plotted:
        print("[INFO] No RT data available to plot KDE.")
        plt.close(fig)
        return

    for res, rt_df in sorted(all_rt_data.items()):
        if not rt_df.empty:
            sns.kdeplot(data=rt_df, x="RT", ax=ax, fill=True, alpha=0.3, linewidth=2,
                        color=COLOR_PALETTE.get(res, '#333333'), label=f"{res}kb")

    ax.set_title("Genome-wide RT Density (Intuitive Model)")
    ax.set_xlabel("Replication Timing (RT)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {output_pdf}")

def plot_10kb_rt_rfd_profile(all_rt_data: Dict[int, pd.DataFrame], output_pdf: Path):
    """Generate a combined RT and RFD profile plot for the 10kb resolution data."""
    print(f"Generating 10kb RT/RFD profile PDF -> {output_pdf}")
    if 10 not in all_rt_data or all_rt_data[10].empty:
        print("[WARN] No 10kb RT data found; skipping 10kb profile PDF.")
        return
        
    df10 = all_rt_data[10]
    with PdfPages(output_pdf) as pdf:
        for chrom in CHROMOSOMES:
            chr_data = df10[df10['chr'] == chrom].sort_values('start')
            if chr_data.empty:
                continue

            fig, ax1 = plt.subplots(figsize=(15, 8))
            ax2 = ax1.twinx()
            
            gpos = chr_data['start'].values / 1e6
            width = (chr_data['end'].values - chr_data['start'].values) / 1e6
            rfd_vals = chr_data['RFD'].values
            
            # Plot RFD
            ax1.bar(gpos, np.abs(rfd_vals - 0.5), width=width, bottom=np.minimum(rfd_vals, 0.5),
                    color=COLOR_RFD, align='edge', label='RFD (10kb)')
            ax1.axhline(0.5, color='k', linestyle='--', linewidth=0.8)
            ax1.set_ylim(-0.1, 1.0)
            ax1.set_ylabel("Replication Fork Direction (RFD)")
            
            # Plot RT
            ax2.plot(gpos, chr_data['RT'].values, color=RT_COLOR, linewidth=2.5, label='RT (10kb)')
            ax2.fill_between(gpos, chr_data['RT'].values, color=RT_COLOR, alpha=0.12)
            ax2.set_ylim(-0.1, 1.05)
            ax2.set_ylabel("Replication Timing (RT)")
            
            # Turn off grid for the right y-axis to avoid overlapping grid lines
            ax2.grid(False)
            
            ax1.set_xlabel("Genomic Position (Mb)")
            fig.suptitle(f"Chromosome {chrom} - 10kb RT & RFD (Intuitive Model)")
            
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc='upper right')
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
    print(f"[OK] Saved {output_pdf}")

# ------------------ Main Execution ------------------

def main():
    """Main function to run the entire RT calculation and plotting pipeline."""
    print("=" * 80)
    print("RT Pipeline (Intuitive Per-Chromosome Algorithm)")
    print("=" * 80)
    
    ensure_directories()
    all_rt_data = process_all_resolutions(RESOLUTIONS)
    
    if not all_rt_data:
        print("\n[ERROR] No RT data was computed for any resolution. Exiting.")
        return

    pdf_multires = PLOT_DIR / f"{OUTPUT_PREFIX}chromosome_profiles_multires_intuitive.pdf"
    pdf_kde = PLOT_DIR / f"{OUTPUT_PREFIX}genome_wide_kde_intuitive.pdf"
    pdf_10kb = PLOT_DIR / f"{OUTPUT_PREFIX}rt_rfd_profile_10kb_intuitive.pdf"

    plot_chromosome_rt_curves_multires(all_rt_data, pdf_multires)
    plot_genome_wide_kde(all_rt_data, pdf_kde)
    plot_10kb_rt_rfd_profile(all_rt_data, pdf_10kb)
    
    print(f"\nAll tasks complete. Check output in {RESULTS_DIR} and {PLOT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()


```






---

## rt_03.5_compare

```python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List
from scipy.stats import pearsonr

# --- Script Configuration ---

# Simplified paths for a public repository.
# Assumes the script is run from the project root, with data in './data/results/'
RESULTS_DIR = Path("./data/results")
PLOT_DIR = Path("./plots")

# Analysis Parameters
RESOLUTIONS: List[int] = [5, 10, 20]
CHROMOSOMES: List[str] = [str(i) for i in range(1, 15)]
METHODS = {
    'integration': 'rt_curve_de',
    'origin_based': 'rt_curve_intuitive'
}

# Column Name Constants
COL_CHR = "chr"
COL_START = "start"
COL_RT = "RT"

# Custom Color Palettes
COLOR_PALETTE_INTEGRATION: Dict[int, str] = {
    5: '#41b349',  # Green
    10: '#2775b6', # Blue
    20: '#fcc307'  # Yellow
}
COLOR_PALETTE_ORIGIN: Dict[int, str] = {
    5: '#add5a2',  # Light Green
    10: '#93d5dc', # Light Blue/Cyan
    20: '#fcc307'  # Yellow
}


def load_all_rt_data() -> Dict[str, Dict[int, pd.DataFrame]]:
    """Loads RT data for all specified methods and resolutions."""
    print("Loading replication timing data...")
    all_data = {method_name: {} for method_name in METHODS}

    for method_name, file_prefix in METHODS.items():
        print(f"\n--- Loading data for '{method_name}' method ---")
        for res in RESOLUTIONS:
            file_path = RESULTS_DIR / f"{file_prefix}_{res}kb.txt"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep='\t', usecols=[COL_CHR, COL_START, COL_RT])
                    df[COL_CHR] = df[COL_CHR].astype(str)
                    all_data[method_name][res] = df
                    print(f"  Successfully loaded {res}kb data from {file_path.name}")
                except Exception as e:
                    print(f"  Error loading {file_path.name}: {e}")
            else:
                print(f"  Warning: File not found, skipping: {file_path.name}")
    
    return all_data


def plot_rt_curve_comparison(all_data: Dict[str, Dict[int, pd.DataFrame]]):
    """Generates a multi-page PDF comparing RT profiles for each chromosome."""
    output_pdf_path = PLOT_DIR / "rt_comparison_chromosome_profiles.pdf"
    print(f"\n--- Generating chromosome-by-chromosome RT profile comparison: {output_pdf_path.name} ---")

    with PdfPages(output_pdf_path) as pdf:
        for chrom in CHROMOSOMES:
            fig, axes = plt.subplots(
                nrows=1, ncols=3, figsize=(24, 6), sharey=True,
                gridspec_kw={'wspace': 0.1}
            )
            fig.suptitle(f'Replication Timing Comparison for Chromosome {chrom}', fontsize=20, y=1.02)

            for i, res in enumerate(RESOLUTIONS):
                ax = axes[i]
                
                # Plot 'Integration' data
                df_integration = all_data['integration'].get(res)
                if df_integration is not None:
                    chr_data = df_integration[df_integration[COL_CHR] == chrom]
                    if not chr_data.empty:
                        ax.plot(
                            chr_data[COL_START] / 1e6, chr_data[COL_RT],
                            color=COLOR_PALETTE_INTEGRATION[res],
                            linewidth=2.5, label='Integration'
                        )
                
                # Plot 'Origin-based' data
                df_origin = all_data['origin_based'].get(res)
                if df_origin is not None:
                    chr_data = df_origin[df_origin[COL_CHR] == chrom]
                    if not chr_data.empty:
                        ax.plot(
                            chr_data[COL_START] / 1e6, chr_data[COL_RT],
                            color=COLOR_PALETTE_ORIGIN[res],
                            linewidth=2.0, linestyle='--', label='Origin-based'
                        )

                ax.set_title(f'{res}kb Resolution', fontsize=14)
                ax.set_xlabel('Genomic Position (Mb)', fontsize=12)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
            
            axes[0].set_ylabel('Replication Timing (RT)', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print("  Chromosome profile comparison PDF saved.")


def plot_kde_comparison(all_data: Dict[str, Dict[int, pd.DataFrame]]):
    """Generates a plot comparing the genome-wide RT density distributions."""
    output_pdf_path = PLOT_DIR / "rt_comparison_genome_wide_kde.pdf"
    print(f"\n--- Generating genome-wide KDE comparison plot: {output_pdf_path.name} ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(22, 6), sharey=True,
        gridspec_kw={'wspace': 0.1}
    )
    fig.suptitle('Genome-wide Replication Timing Density Comparison', fontsize=20, y=1.02)

    for i, res in enumerate(RESOLUTIONS):
        ax = axes[i]
        
        df_integration = all_data['integration'].get(res)
        if df_integration is not None and not df_integration.empty:
            sns.kdeplot(
                data=df_integration, x=COL_RT, ax=ax, fill=True, alpha=0.4,
                color=COLOR_PALETTE_INTEGRATION[res],
                linewidth=2.5, label='Integration'
            )

        df_origin = all_data['origin_based'].get(res)
        if df_origin is not None and not df_origin.empty:
            sns.kdeplot(
                data=df_origin, x=COL_RT, ax=ax, fill=True, alpha=0.5,
                color=COLOR_PALETTE_ORIGIN[res],
                linewidth=2.0, linestyle='--', label='Origin-based'
            )

        ax.set_title(f'{res}kb Resolution', fontsize=14)
        ax.set_xlabel('Replication Timing (RT)', fontsize=12)
        ax.set_xlim(0, 1)
        ax.legend()

    axes[0].set_ylabel('Density', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  KDE comparison PDF saved.")


def plot_10kb_overview_panel(all_data: Dict[str, Dict[int, pd.DataFrame]]):
    """Generates a multi-panel overview figure comparing methods at 10kb resolution."""
    output_pdf_path = PLOT_DIR / "rt_comparison_overview_10kb.pdf"
    print(f"\n--- Generating 10kb resolution overview plot: {output_pdf_path.name} ---")

    df_int = all_data['integration'].get(10)
    df_ori = all_data['origin_based'].get(10)

    if df_int is None or df_ori is None:
        print("  Warning: Missing 10kb data. Skipping overview plot.")
        return

    fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharey=True)
    axes = axes.flatten()

    for i, chrom in enumerate(CHROMOSOMES):
        ax = axes[i]
        
        chr_int = df_int[df_int[COL_CHR] == chrom]
        chr_ori = df_ori[df_ori[COL_CHR] == chrom]
        
        if chr_int.empty or chr_ori.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Chr {chrom}')
            continue
        
        ax.plot(chr_ori[COL_START] / 1e6, chr_ori[COL_RT], color=COLOR_PALETTE_ORIGIN[10], label='Origin-based')
        ax.plot(chr_int[COL_START] / 1e6, chr_int[COL_RT], color=COLOR_PALETTE_INTEGRATION[10], label='Integration')
        
        ax.fill_between(
            chr_int[COL_START] / 1e6, chr_int[COL_RT], chr_ori[COL_RT],
            color='grey', alpha=0.3, interpolate=True, label='Difference'
        )

        merged_df = pd.merge(chr_int, chr_ori, on=COL_START, suffixes=('_int', '_ori')).dropna()
        if len(merged_df) > 1:
            corr, _ = pearsonr(merged_df[f'{COL_RT}_int'], merged_df[f'{COL_RT}_ori'])
            ax.text(0.95, 0.95, f'r = {corr:.3f}', ha='right', va='top', transform=ax.transAxes, fontsize=14, 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', lw=0.5, alpha=0.8))

        ax.set_title(f'Chr {chrom}', fontsize=21)

    # Place legend in the empty 15th panel area
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.91, 0.23), fontsize=18)
    axes[-1].axis('off')

    # Add shared axis labels
    fig.text(0.06, 0.5, 'Replication Timing (RT)', ha='center', va='center', rotation='vertical', fontsize=17)
    fig.text(0.5, 0.04, 'Genomic Position (Mb)', ha='center', va='center', fontsize=17)
    
    plt.tight_layout(rect=[0.02, 0.08, 1, 0.95])
    fig.suptitle('RT Method Comparison Overview - 10kb Bin Size', fontsize=28)
    plt.savefig(output_pdf_path, dpi=300)
    plt.close(fig)
    print("  10kb overview PDF saved.")


def main():
    """Main function to load data and generate comparison plots."""
    PLOT_DIR.mkdir(exist_ok=True)
    
    all_rt_data = load_all_rt_data()

    if not all_rt_data['integration'] and not all_rt_data['origin_based']:
        print("\nNo data could be loaded for either method. Exiting plotting script.")
        return

    print("\n" + "="*50)
    print("      GENERATING COMPARISON VISUALIZATIONS")
    print("="*50)
    
    plot_rt_curve_comparison(all_rt_data)
    plot_kde_comparison(all_rt_data)
    plot_10kb_overview_panel(all_rt_data)
    
    print("\n" + "="*50)
    print("  Comparison plotting completed successfully.")
    print(f"  Plots are saved in: {PLOT_DIR.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()


```






---

## rt_04_virulence_meres_atac

```python


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from pathlib import Path
import re
import seaborn as sns
import warnings
import shutil
import sys

try:
    from PyPDF2 import PdfWriter
except ImportError:
    print("Error: PyPDF2 library not found. Please install it using 'pip install PyPDF2'.")
    sys.exit(1)

warnings.filterwarnings('ignore')

# --- Configuration ---
print("--- Initializing Configuration ---")

# Simplified paths for public repository. Assumes a standard project layout.
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
PLOT_DIR = Path("./plots")

# Input Files
ANNOTATION_FILE = DATA_DIR / "GenesByTaxon_Summary.csv"
CENTROMERE_FILE = DATA_DIR / "pf_centromere.txt"
TELOMERE_FILE = DATA_DIR / "pf_core_region_telomere.tsv"
ATAC_FILE = DATA_DIR / "genome_10kb_t35_signal.bed"
RT_FILE = RESULTS_DIR / "rt_curve_de_10kb.txt" # Assumes this is pre-generated

# Output Files
GENE_LOCATIONS_OUTPUT = RESULTS_DIR / "virulence_gene_locations.csv"
GENE_RT_DETAILS_OUTPUT = RESULTS_DIR / "virulence_gene_rt_details.csv"
BOXPLOT_OUTPUT = PLOT_DIR / "virulence_gene_rt_boxplot.pdf"
GENOME_OVERVIEW_OUTPUT = PLOT_DIR / "pfalciparum_genome_overview.pdf"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Gene families to analyze
GENE_FAMILIES_TO_ANALYZE = {
    'var':    {'symbol': 'var',    'keyword': 'erythrocyte membrane protein 1'},
    'rifin':  {'symbol': 'rif',    'keyword': 'rifin'},
    'stevor': {'symbol': 'stevor', 'keyword': 'stevor'},
    'mc-2tm': {'symbol': 'mc-2tm', 'keyword': 'mc-2tm'}
}

# Plotting Aesthetics
FAMILY_COLORS = {
    'var': '#ff7f0e',    # Orange
    'rifin': '#2ca02c',  # Green
    'stevor': '#9467bd', # Purple
    'mc-2tm': '#FF0000'  # Bright Red
}
RT_COLOR = '#2c7bb6'
CENTROMERE_COLOR = 'black'
SUBTELOMERE_COLOR = '#7f7f7f'
VAR_CLUSTER_COLOR = FAMILY_COLORS['var'] # Match var gene color
ATAC_CMAP = "viridis"

# Chromosome order for plotting
CHROM_ORDER = [f"{i:02d}" for i in range(1, 15)]

# --- Core Functions ---

def parse_location(loc_str):
    """Extracts chromosome, start, and end from a location string."""
    if not isinstance(loc_str, str): return None, None, None
    match = re.search(r'Pf3D7_(\d+)_v3:([\d,]+)..([\d,]+)', loc_str)
    if match:
        return match.group(1).zfill(2), int(match.group(2).replace(',', '')), int(match.group(3).replace(',', ''))
    return None, None, None

def parse_range(range_str):
    """Parses a 'start..end' string into two integers."""
    if pd.isna(range_str) or not isinstance(range_str, str): return None, None
    try:
        start, end = map(int, range_str.split('..'))
        return start, end
    except (ValueError, AttributeError):
        return None, None

def extract_genes_robust(annotation_path, output_path):
    """Extracts virulence gene locations from the main annotation file."""
    print(f"\n🧬 Step 1a: Extracting virulence gene locations from {annotation_path.name}")
    try:
        df = pd.read_csv(annotation_path)
    except FileNotFoundError:
        print(f"ERROR: Annotation file '{annotation_path}' not found.")
        return None

    df.columns = df.columns.str.strip().str.replace('"', '')
    df['symbol_lower'] = df['Gene Name or Symbol'].str.lower()
    df['desc_lower'] = df['Product Description'].str.lower()

    all_families = []
    for family_name, criteria in GENE_FAMILIES_TO_ANALYZE.items():
        symbol_lower, keyword_lower = criteria['symbol'], criteria['keyword']
        condition = (df['symbol_lower'].str.contains(symbol_lower, na=False)) | \
                    (df['desc_lower'].str.contains(keyword_lower, na=False))
        family_df = df[condition].copy()
        family_df['family'] = family_name
        all_families.append(family_df)

    combined = pd.concat(all_families, ignore_index=True).drop_duplicates(subset=['Gene ID'])
    loc_data = combined['Genomic Location (Gene)'].apply(lambda x: pd.Series(parse_location(x), index=['chr', 'start', 'end']))
    combined = pd.concat([combined[['Gene ID', 'family']], loc_data], axis=1).dropna()
    combined[['start', 'end']] = combined[['start', 'end']].astype(int)
    combined.to_csv(output_path, index=False)
    print(f"  -> Success! Generated {output_path.name} with {len(combined)} genes.")
    return combined

def extract_all_named_genes(annotation_path):
    """Extracts all genes with a non-empty symbol/name."""
    print(f"\n🧬 Step 1b: Extracting all named genes from {annotation_path.name}")
    try:
        df = pd.read_csv(annotation_path)
    except FileNotFoundError:
        print(f"ERROR: Annotation file '{annotation_path}' not found.")
        return None

    df.columns = df.columns.str.strip().str.replace('"', '')
    df['is_named'] = df['Gene Name or Symbol'].notna() & (df['Gene Name or Symbol'].str.strip() != '')
    loc_data = df['Genomic Location (Gene)'].apply(lambda x: pd.Series(parse_location(x), index=['chr', 'start', 'end']))
    all_genes = pd.concat([df[['Gene ID', 'is_named']], loc_data], axis=1).dropna(subset=['chr', 'start', 'end'])
    all_genes[['start', 'end']] = all_genes[['start', 'end']].astype(int)
    print(f"  -> Success! Found {len(all_genes)} genes with location info.")
    return all_genes

def calculate_weighted_gene_rt(genes_df, rt_df):
    """Calculates a weighted average RT for each gene based on overlapping RT bins."""
    print("\n🔬 Step 2: Calculating weighted average RT for each gene...")
    rt_values = []
    for _, gene in genes_df.iterrows():
        overlapping_bins = rt_df[(rt_df['chr'] == gene['chr']) & (rt_df['start'] < gene['end']) & (rt_df['end'] > gene['start'])]
        if overlapping_bins.empty:
            rt_values.append(np.nan)
            continue

        total_overlap, weighted_rt_sum = 0, 0
        for _, bin_row in overlapping_bins.iterrows():
            overlap_len = min(gene['end'], bin_row['end']) - max(gene['start'], bin_row['start'])
            if overlap_len > 0:
                total_overlap += overlap_len
                weighted_rt_sum += bin_row['RT'] * overlap_len
        
        final_rt = weighted_rt_sum / total_overlap if total_overlap > 0 else np.nan
        rt_values.append(final_rt)

    genes_df['RT'] = rt_values
    return genes_df.dropna(subset=['RT'])

# --- Plotting Functions ---

def plot_rt_boxplot(virulence_genes_df, all_genes_df, output_path):
    """Generates and saves a boxplot of RT distributions for gene families."""
    print(f"\n🎨 Plotting 1/2: Generating RT distribution boxplot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    named_genes_for_plot = all_genes_df[all_genes_df['is_named']][['RT']].copy()
    named_genes_for_plot['family'] = 'All Named Genes'
    combined_df = pd.concat([virulence_genes_df[['RT', 'family']], named_genes_for_plot])
    combined_df['family'] = combined_df['family'].apply(lambda x: x.capitalize() if x != 'All Named Genes' else x)

    plot_order = [f.capitalize() for f in GENE_FAMILIES_TO_ANALYZE.keys()] + ['All Named Genes']
    plot_palette = list(FAMILY_COLORS.values()) + ['#808080']

    sns.boxplot(data=combined_df, x='family', y='RT', order=plot_order, palette=plot_palette, fliersize=0, ax=ax)
    sns.stripplot(data=combined_df, x='family', y='RT', order=plot_order, color='black', alpha=0.3, jitter=0.2, ax=ax)

    ax.set_title('Replication Timing of P. falciparum Virulence Gene Families', fontsize=18)
    ax.set_ylabel('Replication Timing (0=Late, 1=Early)', fontsize=14)
    ax.set_xlabel('Gene Family', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='x', labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"  -> Boxplot saved to: {output_path}")

def plot_single_chromosome_pdf(chrom, rt_chrom_df, genes_chrom_df, atac_chrom_df, centromere_df, telomere_df, atac_vmin, atac_vmax, output_path):
    """Generates a single-page PDF for one chromosome, combining RT and ATAC plots."""
    fig, (ax_rt, ax_atac) = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={'height_ratios': [16, 1], 'hspace': 0.05})
    chrom_max_len_bp = rt_chrom_df['end'].max()
    chrom_max_len_mb = chrom_max_len_bp / 1e6

    # Panel 1: RT Plot
    ax_rt.plot(rt_chrom_df['start'] / 1e6, rt_chrom_df['RT'], color=RT_COLOR, lw=1.5, zorder=1)
    ax_rt.fill_between(rt_chrom_df['start'] / 1e6, rt_chrom_df['RT'], color=RT_COLOR, alpha=0.2, zorder=0)

    for _, gene in genes_chrom_df.iterrows():
        gene_color = FAMILY_COLORS.get(gene['family'], 'grey')
        ax_rt.axvspan(gene['start'] / 1e6, gene['end'] / 1e6, color=gene_color, alpha=0.3, lw=0, zorder=2)
        ax_rt.plot([gene['start'] / 1e6, gene['end'] / 1e6], [gene['RT'], gene['RT']], color=gene_color, linewidth=4, solid_capstyle='butt', zorder=3)

    # Add genomic features (centromere, telomeres, var clusters)
    feature_y_pos = -0.10
    centro = centromere_df[centromere_df['Chromosome'] == int(chrom)]
    if not centro.empty:
        start_pos, end_pos = centro.iloc[0]['Start position'] / 1e6, centro.iloc[0]['End position'] / 1e6
        ax_rt.plot([start_pos, end_pos], [feature_y_pos, feature_y_pos], color=CENTROMERE_COLOR, linewidth=10, solid_capstyle='butt', zorder=4)

    chrom_info = telomere_df[telomere_df['chr'] == chrom]
    if not chrom_info.empty:
        info = chrom_info.iloc[0]
        if pd.notna(info.get('Core_from_to_start')) and pd.notna(info.get('Core_from_to_end')):
            ax_rt.plot([0, info['Core_from_to_start'] / 1e6], [feature_y_pos, feature_y_pos], color=SUBTELOMERE_COLOR, linewidth=10, solid_capstyle='butt', zorder=4)
            ax_rt.plot([info['Core_from_to_end'] / 1e6, chrom_max_len_mb], [feature_y_pos, feature_y_pos], color=SUBTELOMERE_COLOR, linewidth=10, solid_capstyle='butt', zorder=4)
        for col in ['Internal_var_cluster_1', 'Internal_var_cluster_2']:
            if pd.notna(info.get(f'{col}_start')) and pd.notna(info.get(f'{col}_end')):
                ax_rt.plot([info[f'{col}_start'] / 1e6, info[f'{col}_end'] / 1e6], [feature_y_pos, feature_y_pos], color=VAR_CLUSTER_COLOR, linewidth=10, solid_capstyle='butt', zorder=5)

    # Panel 2: ATAC Bar Plot
    if not atac_chrom_df.empty:
        img = atac_chrom_df['mean_signal'].values[np.newaxis, :]
        ax_atac.imshow(img, aspect='auto', cmap=ATAC_CMAP, vmin=atac_vmin, vmax=atac_vmax, origin='lower', extent=[0, chrom_max_len_mb, 0, 1])

    # Formatting and Labels
    ax_rt.set_title(f"P. falciparum Genome-wide Analysis - Chromosome {chrom}", fontsize=16, pad=15)
    ax_rt.set_xlim(0, chrom_max_len_mb)
    ax_rt.set_ylim(-0.16, 1.05)
    ax_rt.set_ylabel("RT", fontsize=12)
    for spine in ['top', 'right', 'bottom']: ax_rt.spines[spine].set_visible(False)
    
    ax_atac.set_ylabel("ATAC", rotation=0, ha='right', va='center', fontsize=12, labelpad=25)
    ax_atac.get_yaxis().set_ticks([])
    for spine in ['top', 'right']: ax_atac.spines[spine].set_visible(False)
    ax_atac.set_xlabel("Genomic Position (Mb)", fontsize=12)

    # Legends and Colorbar
    legend_elements = [
        mpatches.Patch(color=RT_COLOR, label='Replication Timing', alpha=0.6),
        *[mpatches.Patch(color=c, label=f'{f.capitalize()} Genes', alpha=0.6) for f, c in FAMILY_COLORS.items()],
        Line2D([0], [0], color=CENTROMERE_COLOR, lw=4, label='Centromere'),
        Line2D([0], [0], color=SUBTELOMERE_COLOR, lw=4, label='Subtelomere'),
        Line2D([0], [0], color=VAR_CLUSTER_COLOR, lw=4, label='Var Cluster')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=len(legend_elements), fontsize=9)

    cax = ax_rt.inset_axes([0.9, -0.2, 0.12, 0.05])
    norm = Normalize(vmin=atac_vmin, vmax=atac_vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=ATAC_CMAP, norm=norm), cax=cax, orientation='horizontal')
    cb.set_label('ATAC', size=8)
    cb.set_ticks([atac_vmin, atac_vmax]); cb.set_ticklabels(['Low', 'High'])
    cb.ax.tick_params(labelsize=7)
    
    plt.subplots_adjust(bottom=0.2, top=0.78)
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_genome_overview(rt_df, virulence_genes_df, atac_df, centromere_df, telomere_df, atac_vmin, atac_vmax, output_path):
    """Generates plots for each chromosome and merges them into a single multi-page PDF."""
    print(f"\nPlotting 2/2: Generating and merging 14 chromosome PDFs...")
    temp_dir = PLOT_DIR / "temp_chromosome_pdfs"
    temp_dir.mkdir(exist_ok=True)
    
    pdf_paths = []
    for i, chrom in enumerate(CHROM_ORDER):
        print(f"  -> Generating PDF for Chromosome {chrom} ({i+1}/{len(CHROM_ORDER)})...")
        single_pdf_path = temp_dir / f"chromosome_{chrom}.pdf"
        plot_single_chromosome_pdf(
            chrom,
            rt_df[rt_df['chr'] == chrom],
            virulence_genes_df[virulence_genes_df['chr'] == chrom],
            atac_df[atac_df['chrom'] == f"Pf3D7_{chrom}_v3"].sort_values('start'),
            centromere_df, telomere_df, atac_vmin, atac_vmax, single_pdf_path
        )
        pdf_paths.append(single_pdf_path)

    print(f"  -> Merging {len(pdf_paths)} individual PDFs into '{output_path.name}'...")
    merger = PdfWriter()
    for pdf_path in pdf_paths:
        merger.append(str(pdf_path))
    merger.write(str(output_path))
    merger.close()
    
    shutil.rmtree(temp_dir)
    print(f"  -> Successfully created multi-page PDF: {output_path}")

# --- Main Execution ---

def main():
    """Coordinates the entire analysis and plotting workflow."""
    print("--- Starting P. falciparum RT & ATAC Analysis ---")
    
    # Step 1: Extract gene information
    virulence_genes_df = extract_genes_robust(ANNOTATION_FILE, GENE_LOCATIONS_OUTPUT)
    if virulence_genes_df is None: return

    all_genes_df = extract_all_named_genes(ANNOTATION_FILE)
    if all_genes_df is None: return

    # Step 2: Load and pre-process datasets
    print(f"\n--- Loading and Pre-processing Data ---")
    rt_df = pd.read_csv(RT_FILE, sep='\t')
    rt_df['chr'] = rt_df['chr'].astype(str).str.zfill(2)
    
    centromere_df = pd.read_csv(CENTROMERE_FILE)
    centromere_df.columns = [c.strip() for c in centromere_df.columns]
    centromere_df['Chromosome'] = centromere_df['Chromosome'].str.extract(r'(\d+)').astype(int)
    
    telomere_df = pd.read_csv(TELOMERE_FILE, sep='\t')
    telomere_df.columns = [c.strip() for c in telomere_df.columns]
    telomere_df['chr'] = telomere_df['Chromosome'].str.extract(r'Pf3D7_(\d+)_v3').str.zfill(2)
    for col in ['Core_from_to', 'Internal_var_cluster_1', 'Internal_var_cluster_2']:
        if col in telomere_df.columns:
            coords = telomere_df[col].apply(parse_range)
            telomere_df[f'{col}_start'], telomere_df[f'{col}_end'] = zip(*coords)
    print("  -> RT, Gene, Centromere, and Telomere data loaded successfully.")

    if not ATAC_FILE.exists():
        print(f"ERROR: ATAC file {ATAC_FILE} not found. Exiting.")
        return
    atac_df = pd.read_csv(ATAC_FILE, sep=r"\s+", header=None, names=["chrom", "start", "end", "name", "mean_signal"], engine="python")
    atac_df['mean_signal'] = pd.to_numeric(atac_df['mean_signal'], errors='coerce').fillna(0.0)
    
    # Calculate robust color limits for ATAC data
    sig = atac_df['mean_signal'].values
    atac_vmin, atac_vmax = np.nanpercentile(sig, [1, 99])
    if atac_vmax <= atac_vmin: # Handle cases with low variance
        atac_vmin, atac_vmax = np.min(sig), np.max(sig)
    print("  -> ATAC accessibility data loaded and color scale computed.")

    # Step 3: Calculate RT for genes
    virulence_genes_df['chr'] = virulence_genes_df['chr'].astype(str).str.zfill(2)
    all_genes_df['chr'] = all_genes_df['chr'].astype(str).str.zfill(2)
    virulence_genes_with_rt = calculate_weighted_gene_rt(virulence_genes_df, rt_df)
    all_genes_with_rt = calculate_weighted_gene_rt(all_genes_df, rt_df)

    print(f"\nSaving detailed analysis results to {GENE_RT_DETAILS_OUTPUT}...")
    virulence_genes_with_rt.to_csv(GENE_RT_DETAILS_OUTPUT, index=False)
    print("  -> Done.")

    # Step 4: Generate all plots
    plot_rt_boxplot(virulence_genes_with_rt, all_genes_with_rt, BOXPLOT_OUTPUT)
    plot_genome_overview(rt_df, virulence_genes_with_rt, atac_df, centromere_df, telomere_df,
                         atac_vmin, atac_vmax, GENOME_OVERVIEW_OUTPUT)

    print("\n All tasks complete.")

if __name__ == "__main__":
    main()


```






---

## rt_05_gene_expression_on_rt_

```python

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# --- Configuration ---

# Simplified paths for public code.
# Assumes a structure like:
# your_project/
#  |- data/
#  |   |- gene_RT_vs_expr_t35.tsv
#  |   |- GenesByTaxon_Summary.csv
#  |- plots/
#  |- this_script.py
#
DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True) # Create plots directory if it doesn't exist

# Input and output files
infile = DATA_DIR / "gene_RT_vs_expr_t35.tsv"
family_file = DATA_DIR / "GenesByTaxon_Summary.csv"
output_pdf = PLOTS_DIR / "RT_36hpi_vs_Expr_t35.pdf"

# Colors and gene families to highlight
FAMILY_COLORS = {
    'var': '#ff7f0e',
    'rifin': '#2ca02c',
    'stevor': '#9467bd',
    'mc-2tm': '#d62728',
    'other': 'lightgray'
}
FAMILIES_TO_HIGHLIGHT = ['var', 'rifin', 'stevor', 'mc-2tm']

# --- Data Loading and Processing ---

# Load main dataset and clean it
df = pd.read_csv(infile, sep="\t")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=["RT_t36hpi", "Expr_t35hpi"], inplace=True)
df["RT_t36hpi"] = pd.to_numeric(df["RT_t36hpi"], errors="coerce")
df["Expr_t35hpi"] = pd.to_numeric(df["Expr_t35hpi"], errors="coerce")
df.dropna(subset=["RT_t36hpi", "Expr_t35hpi"], inplace=True)
df['log1p_Expr'] = np.log1p(df["Expr_t35hpi"])

# Function to classify genes into families based on description
def assign_family(description):
    if not isinstance(description, str): return 'other'
    d = description.lower()
    if 'var' in d or 'pfemp1' in d: return 'var'
    if 'rifin' in d: return 'rifin'
    if 'stevor' in d: return 'stevor'
    if 'mc-2tm' in d or '2tm' in d: return 'mc-2tm'
    return 'other'

# Load family data, classify, and merge with the main dataframe
family_df = pd.read_csv(family_file, usecols=["Gene ID", "Product Description"])
family_df.rename(columns={'Gene ID': 'gene_id'}, inplace=True)
family_df['family'] = family_df['Product Description'].apply(assign_family)
df = pd.merge(df, family_df[['gene_id', 'family']], on='gene_id', how='left')
df['family'].fillna('other', inplace=True)

# --- Statistical Analysis ---
pear_r, pear_p = pearsonr(df["RT_t36hpi"], df['log1p_Expr'])
spear_r, spear_p = spearmanr(df["RT_t36hpi"], df['log1p_Expr'])

# --- Plotting ---
sns.set_style("white")
fig = plt.figure(figsize=(10, 8))

# Use GridSpec for a complex layout with a main scatter plot and marginal KDEs
# The top row is made taller (ratio 1.5) to give more space to the KDE plots.
gs = gridspec.GridSpec(
    5, 4, figure=fig,
    hspace=0.06, wspace=0.10,
    height_ratios=[1.5, 1, 1, 1, 1]
)

# Assign axes from the grid
ax_scatter = fig.add_subplot(gs[1:, :]) # Main scatter plot uses the bottom 4 rows
axes_marg_x = {
    fam: fig.add_subplot(gs[0, i], sharex=ax_scatter)
    for i, fam in enumerate(FAMILIES_TO_HIGHLIGHT)
} # One marginal plot for each highlighted family in the top row

# Plot "other" genes as a background
other_df = df[df['family'] == 'other']
ax_scatter.scatter(
    other_df['RT_t36hpi'], other_df['log1p_Expr'],
    s=5, alpha=0.5, color=FAMILY_COLORS['other'], zorder=1
)

# Plot highlighted gene families and their marginal KDEs
for family in FAMILIES_TO_HIGHLIGHT:
    subset_df = df[df['family'] == family]
    if subset_df.empty: continue
    color = FAMILY_COLORS[family]

    # Scatter plot on the main axis
    ax_scatter.scatter(
        subset_df['RT_t36hpi'], subset_df['log1p_Expr'],
        s=10, alpha=0.9, color=color, label=family, zorder=10
    )

    # KDE plot on the corresponding marginal axis
    ax_marg = axes_marg_x[family]
    sns.kdeplot(data=df, x='RT_t36hpi', color='gray', fill=True, alpha=0.35, lw=0.5, ax=ax_marg, bw_adjust=0.85)
    sns.kdeplot(data=subset_df, x='RT_t36hpi', color=color, fill=True, alpha=0.7, lw=1.5, ax=ax_marg, bw_adjust=0.85)
    ax_marg.set_title(family, color=color, fontsize=10, pad=2)
    ax_marg.axis('off')

# Unify the y-axis limits of all marginal KDE plots for consistent scaling
max_ylim = max(ax.get_ylim()[1] for ax in axes_marg_x.values())
for ax in axes_marg_x.values():
    ax.set_ylim(0, max_ylim)

# Format main scatter plot
ax_scatter.set_xlabel("Replication Timing (36 hpi, RT)", fontsize=11)
ax_scatter.set_ylabel("Expression (t35, log1p signal)", fontsize=11)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_scatter.spines[['top', 'right']].set_visible(True)
ax_scatter.set_xlim(-0.05, 1.05)
ax_scatter.set_ylim(-0.5, df['log1p_Expr'].max() * 1.05)

# Add correlation statistics text box
stats_text = f"Pearson r = {pear_r:.3f} (p={pear_p:.1e})\nSpearman ρ = {spear_r:.3f} (p={spear_p:.1e})"
ax_scatter.text(
    0.04, 0.96, stats_text, transform=ax_scatter.transAxes,
    va="top", ha="left", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
)

# Add legend
ax_scatter.legend(loc=(0.035, 0.64), title='Gene Families')
fig.suptitle("Replication Timing (36hpi) vs. Gene Expression (t35)", fontsize=16, y=0.96)

# --- Save Figure ---
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Plot saved to: {output_pdf}")



```






---

## rt_06_gene_families

```python


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import re
import seaborn as sns
import warnings
from pathlib import Path

# Suppress minor warnings for a cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Configuration
# =============================================================================
print("--- Initializing Configuration ---")

# --- Path Configuration ---
# Simplified paths for public code sharing.
# Assumes a directory structure:
# ./
# |- data/
# |- results/
# |- plots/
# |- your_script.py
RESULTS_DIR = Path("./results")
PLOT_DIR = Path("./plots")
DATA_DIR = Path("./data")
ANNOTATION_FILE = DATA_DIR / "GenesByTaxon_Summary.csv"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- Analysis Parameters ---
# Use precomputed RT data for all genes
ALL_GENES_RT_INPUT = RESULTS_DIR / "rt_all_genes_rt_detail.csv"
# Define the output PDF file for the violin plot
SUBSET_VIOLIN_PDF = PLOT_DIR / "rt_subset_families_violinplot.pdf"

# Minimum number of genes required to include a family in the plot
MIN_GENE_COUNT = 1

# --- Define Gene Modules for Plotting ---
# These three groups will be plotted in separate panels.
SUBSET_MODULES = {
    "Immune Escape Module": {
        "families": ["VAR", "RIFIN", "stevor", "MC-2TM"],
        "description": "Virulence families for immune evasion"
    },
    "Translation Machinery Module": {
        "families": ["rRNA", "RPS", "RPL", "tRNA", "snRNA"],
        "description": "Core translation-related families"
    },
    "Invasion-Structure Module": {
        "families": ["RAP protein", "IMC", "PSOP"],
        "description": "Life cycle (invasion/structure) families"
    }
}

# =============================================================================
# 2. Utility Functions for Family Assignment
# =============================================================================
# Generic terms to be removed from gene descriptions for cleaner family names
GENERIC_TERMS = [
    'putative', 'protein', 'family', 'containing', 'domain', 'exported',
    'conserved', 'surface', 'antigen', 'membrane', 'hypothetical',
    'unknown function', 'unknown', 'fragment', 'pseudogene', 'subunit',
    'binding', 'interacting', 'small'
]
GENERIC_PATTERN = re.compile(r'\b(?:' + r'|'.join([re.escape(x) for x in GENERIC_TERMS]) + r')\b', flags=re.I)

def extract_symbol_prefix(symbol):
    """Extracts a potential family prefix from a gene symbol."""
    SPECIAL_PREFIXES = ['MC-2TM']
    if not isinstance(symbol, str): return ''
    s = symbol.strip()
    if not s or s.lower() in ('n/a', '-', ''): return ''

    for prefix in SPECIAL_PREFIXES:
        if s.lower().startswith(prefix.lower()):
            return prefix.lower()

    match = re.match(r'^([A-Za-z]{2,})', s.split()[0])
    return match.group(1).lower() if match else ''

def assign_families_simplified(df, min_sym_count=2):
    """Assigns genes to families using a custom, multi-pass logic."""
    print(" -> Assigning gene families...")

    KEYWORD_FAMILIES = [
        'RAP protein', 'AAA family ATPase', 'ADP-ribosylation factor',
        'ribosomal RNA', 'CPW-WPC family protein', 'mitochondrial carrier protein',
        'kinesin', 'HSP20-like chaperone', 'proteasome subunit',
        'regulator of chromosome condensation', 'small nucleolar RNA',
        'ankyrin-repeat protein', 'stevor', 'tRNA',
        'ubiquitin carboxyl-terminal hydrolase', 'ubiquitin-conjugating enzyme',
        'MC-2TM', 'IMC', 'PSOP', 'RPS', 'RPL'
    ]

    fam = pd.Series('', index=df.index, dtype=object)
    sym = df['symbol_lower']
    desc = df['desc_lower']

    # Pass 0: Hard-coded families (VAR, RIFIN)
    var_mask = sym.str.contains(r'\bvar\b', na=False) | desc.str.contains('erythrocyte membrane protein 1', na=False)
    fam.loc[var_mask] = 'VAR'

    rifin_mask = sym.str.contains(r'\brif\b', na=False) | desc.str.contains('rifin', na=False)
    fam.loc[rifin_mask & ~var_mask] = 'RIFIN'
    is_assigned = fam != ''
    print(f"    -> Pass 0 (Special): Assigned {is_assigned.sum()} genes (VAR/RIFIN).")

    # Pass 1: From gene symbol prefixes
    unassigned = ~is_assigned
    symbol_rows = unassigned & (sym != 'n/a')
    if symbol_rows.any():
        sym_prefix = sym[symbol_rows].apply(extract_symbol_prefix)
        prefix_counts = sym_prefix.value_counts()
        good_prefixes = set(prefix_counts[prefix_counts >= min_sym_count].index)
        for idx, p in sym_prefix.items():
            if p and p in good_prefixes:
                fam.at[idx] = p.upper()

    print(f"    -> Pass 1 (Symbols): Assigned {(fam != '').sum() - is_assigned.sum()} new genes.")
    is_assigned = fam != ''

    # Pass 2: Keyword search in descriptions for genes with N/A symbols
    unassigned = ~is_assigned
    na_symbol_rows = unassigned & (sym == 'n/a')
    pass2_start_count = is_assigned.sum()
    if na_symbol_rows.any():
        for keyword in KEYWORD_FAMILIES:
            current_unassigned = na_symbol_rows & ~is_assigned
            if not current_unassigned.any(): break
            
            # Search only in the first part of the description before a comma
            desc_to_search = desc[current_unassigned].str.split(',').str[0]
            match_mask = desc_to_search.str.contains(r'\b' + re.escape(keyword.lower()) + r'\b', na=False, flags=re.IGNORECASE)
            
            if match_mask.any():
                match_indices = desc_to_search[match_mask].index
                fam.loc[match_indices] = keyword
                is_assigned.loc[match_indices] = True

    print(f"    -> Pass 2 (Keywords): Assigned {is_assigned.sum() - pass2_start_count} new genes.")

    fam.replace('', 'other', inplace=True)
    df['family'] = fam
    print(f" -> Family assignment complete. Total genes processed: {len(df)}.")
    return df

# =============================================================================
# 3. Data Loading and Preparation
# =============================================================================
def load_precomputed_rt_data(rt_input_path):
    """Loads precomputed RT data from the specified CSV file."""
    print(f"\n Loading precomputed RT data from {rt_input_path}")
    if not rt_input_path.exists():
        print(f"ERROR: Precomputed RT file not found at {rt_input_path}")
        return None

    df = pd.read_csv(rt_input_path)
    required_cols = ['Gene ID', 'family', 'chr', 'start', 'end', 'RT']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing one or more required columns in RT file. Needed: {required_cols}")
        return None

    # Filter for valid RT values (between 0 and 1)
    df.dropna(subset=['RT'], inplace=True)
    df = df[df['RT'].between(0, 1)]  # RT scale: 0=late, 1=early
    print(f" -> Loaded {len(df)} genes with valid RT values.")
    return df

def filter_subset_data(rt_df, subset_modules):
    """Filters the main RT DataFrame to include only families in the defined modules."""
    print("\n Filtering data for specified modules...")
    all_subset_families = {fam for module in subset_modules.values() for fam in module['families']}
    existing_families = set(rt_df['family'].unique())
    
    # Report any defined families that are not found in the dataset
    missing_families = all_subset_families - existing_families
    if missing_families:
        print(f" Warning: The following families were not found in the data: {', '.join(missing_families)}")

    # Build the subset DataFrame
    subset_data_frames = []
    for module_name, module_info in subset_modules.items():
        module_families = [fam for fam in module_info['families'] if fam in existing_families]
        if not module_families:
            print(f" Warning: No valid families found for module: {module_name}")
            continue
        
        module_df = rt_df[rt_df['family'].isin(module_families)].copy()
        module_df['module'] = module_name  # Add module label for plotting
        subset_data_frames.append(module_df)
        print(f" -> Found {len(module_df)} genes for '{module_name}' ({', '.join(module_families)})")

    if not subset_data_frames:
        print(" ERROR: No data found for any of the specified modules.")
        return None
    
    return pd.concat(subset_data_frames, ignore_index=True)

# =============================================================================
# 4. Plotting Function
# =============================================================================
def plot_subset_violin(subset_df, full_rt_df, output_path, subset_modules):
    """Generates a 3-panel horizontal violin plot for the specified gene modules."""
    print(f"\n🎨 Generating 3-panel subset violin plot...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    fig.suptitle('Replication Timing (RT) of Gene Modules in P. falciparum',
                 fontsize=28, fontweight='bold', y=0.99)

    palette = sns.color_palette('viridis', n_colors=max(len(m['families']) for m in subset_modules.values()))
    
    # Calculate the genome-wide mean RT using the full dataset for an accurate reference line
    genome_mean_rt = full_rt_df['RT'].mean()

    # Plot each module in a separate panel
    for idx, (module_name, module_info) in enumerate(subset_modules.items()):
        ax = axes[idx]
        module_data = subset_df[subset_df['module'] == module_name]

        if module_data.empty:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=12, style='italic', transform=ax.transAxes)
            ax.set_title(module_name, fontsize=20, fontweight='bold')
            ax.set_xlabel('Replication Timing (RT)', fontsize=12)
            continue
        
        # Filter families by the minimum gene count
        family_counts = module_data['family'].value_counts()
        valid_families = family_counts[family_counts >= MIN_GENE_COUNT].index.tolist()
        
        if not valid_families:
            ax.text(0.5, 0.5, f'No families with ≥ {MIN_GENE_COUNT} genes', ha='center', va='center', fontsize=11, style='italic', transform=ax.transAxes)
            ax.set_title(module_name, fontsize=22, fontweight='bold')
            continue
            
        # Sort families by mean RT (ascending, so earlier families are on the left)
        mean_rt_per_fam = module_data.groupby('family')['RT'].mean().loc[valid_families].sort_values()
        sorted_families = mean_rt_per_fam.index.tolist()
        
        sns.violinplot(
            data=module_data[module_data['family'].isin(sorted_families)],
            x='family', y='RT', order=sorted_families,
            palette=palette[:len(sorted_families)],
            inner='quartile', ax=ax
        )
        
        # Add the genome-wide mean RT line for reference
        ax.axhline(y=genome_mean_rt, color='red', linestyle='--', linewidth=1.5,
                   label=f'Genome Mean RT: {genome_mean_rt:.2f}')
        
        ax.set_title(module_name, fontsize=22, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('Replication Timing (RT)' if idx == 0 else '', fontsize=14)
        
        # Create custom x-tick labels with family name and gene count
        xtick_labels = [f"{fam}\n(n={family_counts[fam]})" for fam in sorted_families]
        ax.set_xticklabels(xtick_labels, ha='center', fontsize=18)
        
        # Annotate each violin with its mean RT value
        for i, fam in enumerate(sorted_families):
            fam_mean = mean_rt_per_fam[fam]
            fam_median = module_data[module_data['family'] == fam]['RT'].median()
            text = ax.text(i, fam_median, f'{fam_mean:.2f}', ha='center', va='center', 
                           color='white', fontsize=18, fontweight='bold')
            text.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground='black')])
        
        if idx == 2:
            ax.legend(loc='upper right', fontsize=18)

        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Subset violin plot saved to: {output_path}")

# =============================================================================
# 5. Main Execution
# =============================================================================
def main():
    """Main function to execute the analysis pipeline."""
    print("--- Starting P. falciparum Gene Module RT Analysis ---")

    # Step 1: Load the precomputed RT data for all genes
    full_rt_df = load_precomputed_rt_data(ALL_GENES_RT_INPUT)
    if full_rt_df is None:
        print("Exiting: Failed to load RT data.")
        return

    # Step 2: Filter the data to include only the specified subset modules
    subset_df = filter_subset_data(full_rt_df, SUBSET_MODULES)
    if subset_df is None:
        print("Exiting: No valid data found for the specified modules.")
        return

    # Step 3: Generate the 3-panel violin plot
    # Pass the full dataset to the plotting function to calculate the global mean RT
    plot_subset_violin(subset_df, full_rt_df, SUBSET_VIOLIN_PDF, SUBSET_MODULES)

    print(f"\nAnalysis complete! Output file is available at: {SUBSET_VIOLIN_PDF}")

if __name__ == "__main__":
    main()


```






---

## rt_07_homolog

```python


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib as mpl

# Ensure fonts are embedded correctly in PDF outputs
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams["axes.unicode_minus"] = False
sns.set(style="whitegrid")

class RTAnalyzer:
    """
    Analyzes and compares replication timing (RT) data between
    Plasmodium falciparum (Pf) and Human (Hs) homologous genes.
    """
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.hs_cell_lines = ['H1', 'H9']
        self.window_size_1kb = 1000
        self.window_size_10kb = 10000

        # Data containers
        self.df_pf_10kb = None
        self.df_hs_1kb = {}
        self.df_hs_10kb = {}
        self.df_homolog = None
        self.df_final = None

    def load_pf_rt_data(self, file_path="data/pf_rt_10kb.txt"):
        """Load Pf 10kb RT data."""
        full_path = os.path.join(self.base_dir, file_path)
        print(f"Loading Plasmodium falciparum RT (10kb) from: {full_path}")
        self.df_pf_10kb = pd.read_csv(full_path, sep="\t", engine="python")
        if 'chr' not in self.df_pf_10kb.columns:
            raise ValueError("Pf RT file must contain a 'chr' column.")
        self.df_pf_10kb["chr_std"] = self.df_pf_10kb["chr"].astype(str).str.replace(r"^chr", "", regex=True).str.lstrip("0")
        print(f"Loaded Pf RT: {self.df_pf_10kb.shape[0]} rows")

    def load_human_rt_data(self, data_dir="data/human_rt"):
        """Load human 1kb RT files and build per-cell-line DataFrames."""
        full_dir = os.path.join(self.base_dir, data_dir)
        print(f"Loading human 1kb RT files from: {full_dir}")
        if not os.path.isdir(full_dir):
            raise FileNotFoundError(f"Human RT directory not found: {full_dir}")

        for cell_line in self.hs_cell_lines:
            file_pattern = re.compile(rf'^time_data_{cell_line}_chr\[(.*?)\]\.txt$')
            rt_files = []
            for filename in os.listdir(full_dir):
                m = file_pattern.match(filename)
                if m:
                    chr_num = m.group(1)
                    rt_files.append((chr_num, os.path.join(full_dir, filename)))

            rt_files = sorted(rt_files, key=lambda x: x[0])
            data_rows = []
            if not rt_files:
                print(f"Warning: no RT files found for {cell_line} in {full_dir}.")
            for chr_num, file_path in tqdm(rt_files, desc=f"Processing {cell_line}"):
                with open(file_path, 'r') as fh:
                    rt_values = [float(ln) for ln in fh if ln.strip().replace('.', '', 1).isdigit()]
                for i, rt in enumerate(rt_values):
                    start = i * self.window_size_1kb
                    end = start + self.window_size_1kb
                    data_rows.append({
                        "cell_line": cell_line, "chr": str(chr_num).lstrip("0"),
                        "start": start, "end": end, "rt_1kb": rt
                    })
            df_1kb = pd.DataFrame(data_rows)
            self.df_hs_1kb[cell_line] = df_1kb
            print(f"{cell_line} 1kb rows: {len(df_1kb)}")

    def aggregate_human_to_10kb(self):
        """Aggregate 1kb human RT to 10kb bins by mean."""
        for cell_line in self.hs_cell_lines:
            df = self.df_hs_1kb.get(cell_line)
            if df is None or df.empty:
                print(f"Warning: no 1kb data for {cell_line}, skipping aggregation.")
                self.df_hs_10kb[cell_line] = pd.DataFrame()
                continue
            df = df.copy()
            df["start_10kb"] = (df["start"] // self.window_size_10kb) * self.window_size_10kb
            df_agg = df.groupby(["cell_line", "chr", "start_10kb"])["rt_1kb"].mean().reset_index()
            df_agg = df_agg.rename(columns={"start_10kb": "start", "rt_1kb": "rt_10kb"})
            df_agg["end"] = df_agg["start"] + self.window_size_10kb
            self.df_hs_10kb[cell_line] = df_agg
            print(f"{cell_line} aggregated to 10kb: {len(df_agg)} rows")

    def normalize_human_rt(self):
        """Normalize human 10kb RT to a 0-1 scale (min-max)."""
        for cell_line in self.hs_cell_lines:
            df = self.df_hs_10kb.get(cell_line)
            if df is None or df.empty:
                print(f"Warning: no 10kb data for {cell_line} to normalize.")
                continue
            df = df.copy()
            min_rt, max_rt = df["rt_10kb"].min(), df["rt_10kb"].max()
            if pd.isna(min_rt) or pd.isna(max_rt) or (max_rt - min_rt) == 0:
                df["rt_10kb_norm"] = np.nan
            else:
                df["rt_10kb_norm"] = (df["rt_10kb"] - min_rt) / (max_rt - min_rt)
            self.df_hs_10kb[cell_line] = df
            print(f"{cell_line} normalized to 0-1 (min={min_rt:.2f}, max={max_rt:.2f})")

    def load_homolog_data(self, file_path="data/homolog_report.csv"):
        """Load homolog report."""
        full_path = os.path.join(self.base_dir, file_path)
        print(f"Loading homolog report from: {full_path}")
        self.df_homolog = pd.read_csv(full_path, low_memory=False)
        self.df_homolog["Pf_chr_std"] = self.df_homolog["Pf_chr"].astype(str).str.replace(r"^chr", "", regex=True).str.lstrip("0")
        self.df_homolog["Hs_chr_std"] = self.df_homolog["Hs_chr"].astype(str).str.replace(r"^chr", "", regex=True).str.lstrip("0")
        print(f"Loaded homologs: {len(self.df_homolog)} rows")

    def calculate_weighted_rt(self, gene_start, gene_end, rt_df):
        """Compute weighted average RT for a gene based on overlapping RT intervals."""
        if rt_df is None or rt_df.empty or pd.isna(gene_start) or pd.isna(gene_end):
            return np.nan
        total_len, weighted_sum = 0.0, 0.0
        for _, row in rt_df.iterrows():
            overlap = max(0, min(gene_end, row["end"]) - max(gene_start, row["start"]))
            if overlap > 0:
                total_len += overlap
                weighted_sum += overlap * row["rt"]
        return weighted_sum / total_len if total_len > 0 else np.nan

    def match_homolog_with_rt(self):
        """Map homolog pairs to their corresponding weighted Pf and Human RT values."""
        print("Mapping homologs to RT bins...")
        if self.df_homolog is None:
            raise ValueError("No homolog data loaded.")
        results = []
        for _, row in tqdm(self.df_homolog.iterrows(), total=len(self.df_homolog), desc="Matching genes to RT"):
            pf_rt_chr = self.df_pf_10kb[self.df_pf_10kb["chr_std"] == str(row.get("Pf_chr_std"))]
            pf_rt_val = self.calculate_weighted_rt(row.get("Pf_start"), row.get("Pf_end"), pf_rt_chr[["start", "end", "RT"]].rename(columns={"RT": "rt"}))
            
            hs_vals = {}
            for cell_line in self.hs_cell_lines:
                df_hs_bins = self.df_hs_10kb.get(cell_line, pd.DataFrame())
                df_chr = df_hs_bins[df_hs_bins["chr"].astype(str) == str(row.get("Hs_chr_std"))]
                if not df_chr.empty and "rt_10kb_norm" in df_chr.columns:
                    hs_vals[cell_line] = self.calculate_weighted_rt(row.get("Hs_start"), row.get("Hs_end"), df_chr[["start", "end", "rt_10kb_norm"]].rename(columns={"rt_10kb_norm": "rt"}))
                else:
                    hs_vals[cell_line] = np.nan
            
            res = row.to_dict()
            res.update({"Pf_RT": pf_rt_val, "Hs_H1_RT": hs_vals.get("H1"), "Hs_H9_RT": hs_vals.get("H9")})
            results.append(res)

        df_results = pd.DataFrame(results)
        df_results = df_results.dropna(subset=["Pf_RT", "Hs_H1_RT", "Hs_H9_RT"])
        if not df_results.empty:
            df_results["Hs_H1H9_RT"] = df_results[["Hs_H1_RT", "Hs_H9_RT"]].mean(axis=1)
        
        self.df_final = df_results
        out_path = os.path.join(self.base_dir, "output/homolog_rt_matched.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.df_final.to_csv(out_path, index=False)
        print(f"Mapping complete. Saved matched table to: {out_path} ({len(self.df_final)} rows)")

    def calculate_correlation(self):
        """Compute Pearson correlations between Pf and Human RTs."""
        if self.df_final is None or self.df_final.empty:
            print("No final mapped data to compute correlations.")
            return {}
        res = {}
        pairs = {"H1": "Hs_H1_RT", "H9": "Hs_H9_RT", "H1H9": "Hs_H1H9_RT"}
        for name, col in pairs.items():
            try:
                r, p = pearsonr(self.df_final["Pf_RT"], self.df_final[col])
                res[name] = {"r": r, "p": p}
            except Exception:
                res[name] = {"r": np.nan, "p": np.nan}
        print("Pearson correlations: ", res)
        return res

    def find_outliers(self, human_col="Hs_H1H9_RT", human_thresh=0.4, delta_thresh=0.2,
                      require_both_h1h9=False, sort_by="delta", top_n=None, out_csv=None):
        """
        Find outliers with high human RT values or large deviations from Pf RT.
        
        Args:
            human_col: Column for human RT ('Hs_H1_RT', 'Hs_H9_RT', or 'Hs_H1H9_RT').
            human_thresh: Minimum human RT value to be considered a candidate.
            delta_thresh: Minimum difference (human - Pf) to be considered an outlier.
            require_both_h1h9: If True, both H1 and H9 must meet the threshold.
            sort_by: 'delta', human_col, or 'Pf_RT' to sort the results.
            top_n: If set, return only the top N results.
            out_csv: Path to save the filtered outliers as a CSV file.
        Returns:
            A DataFrame of the filtered and sorted outliers.
        """
        if self.df_final is None or self.df_final.empty:
            raise ValueError("No matched data available (df_final is empty).")

        df = self.df_final.copy()
        df["delta_H1"] = df["Hs_H1_RT"] - df["Pf_RT"]
        df["delta_H9"] = df["Hs_H9_RT"] - df["Pf_RT"]
        df["delta_H1H9"] = df["Hs_H1H9_RT"] - df["Pf_RT"]
        
        delta_map = {"Hs_H1_RT": "delta_H1", "Hs_H9_RT": "delta_H9", "Hs_H1H9_RT": "delta_H1H9"}
        delta_col = delta_map.get(human_col)
        if delta_col is None:
            raise ValueError(f"Unknown human_col: {human_col}")

        mask = df[human_col] > human_thresh
        if require_both_h1h9:
            mask &= (df["Hs_H1_RT"] > human_thresh) & (df["Hs_H9_RT"] > human_thresh)
        if delta_thresh is not None and delta_thresh > 0:
            mask &= (df[delta_col] >= delta_thresh)

        out = df[mask].copy()
        
        sort_col = delta_col if sort_by == "delta" else sort_by
        out = out.sort_values(sort_col, ascending=False)
        
        if top_n:
            out = out.head(top_n)

        if out_csv:
            full_out_path = os.path.join(self.base_dir, out_csv)
            os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
            out.to_csv(full_out_path, index=False)
            print(f"Saved {len(out)} outliers to {full_out_path}")
        
        print(f"Found {len(out)} outliers based on specified criteria.")
        return out

    def plot_scatter_with_highlights(self, output_file="output/rt_pf_hs_highlights.pdf",
                                     human_thresh=0.4, delta_thresh=0.2,
                                     top_n_per_panel=5, label_field="Hs_gene_name"):
        """
        Plot a 3-panel scatter, highlighting the top N outliers in each panel.
        """
        if self.df_final is None or self.df_final.empty:
            print("No matched data to plot.")
            return

        df = self.df_final.copy()
        df["delta_H1"] = df["Hs_H1_RT"] - df["Pf_RT"]
        df["delta_H9"] = df["Hs_H9_RT"] - df["Pf_RT"]
        df["delta_H1H9"] = df["Hs_H1H9_RT"] - df["Pf_RT"]

        panels = [("Hs_H1_RT", "Pf vs H1", "delta_H1"),
                  ("Hs_H9_RT", "Pf vs H9", "delta_H9"),
                  ("Hs_H1H9_RT", "Pf vs H1+H9 (mean)", "delta_H1H9")]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle('Pf RT vs Human Homolog RT', fontsize=23, fontweight='bold')
        for ax, (col, title, delta_col) in zip(axes, panels):
            ax.scatter(df["Pf_RT"], df[col], s=18, alpha=0.4, edgecolor="none", rasterized=True)

            cand = df[(df[col] > human_thresh) & (df[delta_col] >= delta_thresh)]
            cand = cand.sort_values(delta_col, ascending=False).head(top_n_per_panel)

            for _, row in cand.iterrows():
                label = row.get(label_field, "")
                if pd.notna(label):
                    ax.annotate(str(label), xy=(row["Pf_RT"], row[col]), xytext=(5, 5),
                                textcoords='offset points', fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6),
                                arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.6))
            
            r, p = pearsonr(df["Pf_RT"].fillna(0), df[col].fillna(0))
            ax.text(0.03, 0.88, f"Pearson R = {r:.3f}\np = {p:.3g}\nn = {len(df)}",
                    transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.8))
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Pf RT")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.45)

        axes[0].set_ylabel("Human RT (normalized 0-1)")
        plt.tight_layout()
        out_path = os.path.join(self.base_dir, output_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved highlighted scatter plot to: {out_path}")

    def plot_scatter_three_panels(self, output_file="output/rt_pf_hs.pdf"):
        """Create a 3-panel scatter plot comparing Pf and Human RTs."""
        if self.df_final is None or self.df_final.empty:
            print("No matched data to plot.")
            return
        correlations = self.calculate_correlation()
        panels = [("Hs_H1_RT", "Pf vs H1"), ("Hs_H9_RT", "Pf vs H9"), ("Hs_H1H9_RT", "Pf vs H1+H9 (mean)")]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for ax, (col, title) in zip(axes, panels):
            ax.scatter(self.df_final["Pf_RT"], self.df_final[col], s=40, alpha=0.6, edgecolor="none", rasterized=True)

            key = "H1H9" if "H1H9" in col else col.split("_")[1]
            corr = correlations.get(key, {"r": np.nan, "p": np.nan})
            ax.text(0.03, 0.84, f"Pearson R = {corr['r']:.3f}\np = {corr['p']:.4f}\nn = {len(self.df_final)}",
                    transform=ax.transAxes, fontsize=16, bbox=dict(facecolor="white", alpha=0.8))
            ax.set_title(title, fontsize=18)
            ax.set_xlabel("Pf RT", fontsize=12)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.6)

        axes[0].set_ylabel("Human RT (normalized 0-1)", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(self.base_dir, output_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved 3-panel RT comparison plot to: {out_path}")

    def run_pipeline(self):
        """Run the full analysis pipeline end-to-end."""
        print("Starting RT comparison pipeline...")
        self.load_pf_rt_data()
        self.load_human_rt_data()
        self.aggregate_human_to_10kb()
        self.normalize_human_rt()
        self.load_homolog_data()
        self.match_homolog_with_rt()
        self.calculate_correlation()
        self.plot_scatter_three_panels()
        self.plot_scatter_with_highlights()
        print("Pipeline finished.")


def create_dummy_data(base_dir="."):
    """Generates dummy data files for demonstration purposes."""
    print("Creating dummy data for demonstration...")
    # Define simplified paths
    data_path = os.path.join(base_dir, "data")
    human_rt_path = os.path.join(data_path, "human_rt")
    output_path = os.path.join(base_dir, "output")
    os.makedirs(human_rt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Dummy Pf RT data
    pf_rt_file = os.path.join(data_path, "pf_rt_10kb.txt")
    if not os.path.exists(pf_rt_file):
        pf_data = {'chr': ['1']*100, 'start': range(0, 1000000, 10000), 'end': range(10000, 1000001, 10000), 'RT': np.random.rand(100)}
        pd.DataFrame(pf_data).to_csv(pf_rt_file, sep='\t', index=False)

    # Dummy Human RT data
    for cl in ['H1', 'H9']:
        for ch_num in range(1, 3):
            human_rt_file = os.path.join(human_rt_path, f"time_data_{cl}_chr[{ch_num}].txt")
            if not os.path.exists(human_rt_file):
                with open(human_rt_file, "w") as f:
                    for _ in range(1000):
                        f.write(f"{np.random.rand()}\n")

    # Dummy Homolog data
    homolog_file = os.path.join(data_path, "homolog_report.csv")
    if not os.path.exists(homolog_file):
        starts_pf = np.random.randint(0, 900000, 50)
        starts_hs = np.random.randint(0, 9000, 50)
        homolog_data = {
            'Pf_gene_id': [f'PF_G{i}' for i in range(50)], 'Pf_chr': ['1']*50,
            'Pf_start': starts_pf, 'Pf_end': starts_pf + 1000,
            'Human_gene_id': [f'HS_G{i}' for i in range(50)], 'Hs_chr': ['1']*25 + ['2']*25,
            'Hs_start': starts_hs, 'Hs_end': starts_hs + 1000,
            'Pf_gene_name': [f'pf_gene_{i}' for i in range(50)],
            'Human_gene_name': [f'hs_gene_{i}' for i in range(50)]
        }
        pd.DataFrame(homolog_data).to_csv(homolog_file, index=False)
    print("Dummy data created.")


if __name__ == "__main__":
    base_directory = "."
    create_dummy_data(base_directory)
    
    analyzer = RTAnalyzer(base_directory)
    
    # Option 1: Run the full pipeline
    # analyzer.run_pipeline()

    # Option 2: Run steps individually to generate plots
    analyzer.load_pf_rt_data()
    analyzer.load_human_rt_data()
    analyzer.aggregate_human_to_10kb()
    analyzer.normalize_human_rt()
    analyzer.load_homolog_data()
    analyzer.match_homolog_with_rt()

    # Find and save some outliers
    analyzer.find_outliers(out_csv="output/outliers_report.csv", top_n=20)
    
    # Generate the plots with simplified output paths
    analyzer.plot_scatter_three_panels()
    analyzer.plot_scatter_with_highlights()
    
    print("\nAnalysis complete. Check the 'output' directory for results.")



```






---

## rt_07.5_scatter

```python


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Ensure PDF fonts are editable for post-processing in vector graphics software
mpl.rcParams['pdf.fonttype'] = 42

# =============================================================================
# Configuration
# =============================================================================
# Assumes the following directory structure:
# ./
# |- results/
# |  |- homolog_rt_matched.csv  (Input file)
# |- plots/
# |  |- convergent_divergent/ (Output directory)
# |- your_script.py

INPUT_FILE = "results/homolog_rt_matched.csv"
OUTPUT_DIR = "plots/convergent_divergent"

# Define thresholds for classifying gene behavior
CONVERGENT_THRESHOLD = 0.15
DIVERGENT_THRESHOLD = 0.20

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Validate that essential columns are present
required_cols = ["Pf_RT", "Hs_H1_RT", "Hs_H9_RT"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col} in {INPUT_FILE}")

# Calculate RT differences (delta) between species and within human cell lines
df["delta_H1"] = df["Hs_H1_RT"] - df["Pf_RT"]
df["delta_H9"] = df["Hs_H9_RT"] - df["Pf_RT"]
df["hs_internal_diff_abs"] = (df["Hs_H1_RT"] - df["Hs_H9_RT"]).abs()
df["pf_diff_mean"] = (df["delta_H1"].abs() + df["delta_H9"].abs()) / 2
df["pf_diff_max"] = df[["delta_H1", "delta_H9"]].abs().max(axis=1)

# Pre-calculate boolean flags for divergence classification
df["H1_earlier"] = df["delta_H1"] > DIVERGENT_THRESHOLD
df["H9_earlier"] = df["delta_H9"] > DIVERGENT_THRESHOLD
df["H1_later"] = df["delta_H1"] < -DIVERGENT_THRESHOLD
df["H9_later"] = df["delta_H9"] < -DIVERGENT_THRESHOLD

def classify_gene_behavior(row):
    """Classifies each gene into a convergent, divergent, or intermediate category."""
    if row["pf_diff_mean"] <= CONVERGENT_THRESHOLD:
        if abs(row["delta_H1"]) <= CONVERGENT_THRESHOLD and abs(row["delta_H9"]) <= CONVERGENT_THRESHOLD:
            return "Convergent (both)"
        elif abs(row["delta_H1"]) <= CONVERGENT_THRESHOLD:
            return "Convergent (H1)"
        else: # abs(row["delta_H9"]) <= CONVERGENT_THRESHOLD
            return "Convergent (H9)"

    if row["pf_diff_max"] >= DIVERGENT_THRESHOLD:
        if (row["H1_earlier"] and row["H9_later"]) or (row["H9_earlier"] and row["H1_later"]):
            return "Divergent (opposite)"
        if row["H1_earlier"] and row["H9_earlier"]:
            return "Divergent (both earlier)"
        if row["H1_later"] and row["H9_later"]:
            return "Divergent (both later)"
        if row["H1_earlier"]: return "Divergent (H1 earlier)"
        if row["H9_earlier"]: return "Divergent (H9 earlier)"
        if row["H1_later"]: return "Divergent (H1 later)"
        if row["H9_later"]: return "Divergent (H9 later)"
        return "Divergent (other)"
        
    return "Intermediate"

df["category"] = df.apply(classify_gene_behavior, axis=1)

# =============================================================================
# Color Scheme
# =============================================================================
COLOR_MAP = {
    "Convergent (both)": "#219ebc",
    "Convergent (H1)": "#8ecae6",
    "Convergent (H9)": "#023047",
    "Divergent (opposite)": "#ffb703",
    "Divergent (both earlier)": "#2a9d8f",
    "Divergent (H1 earlier)": "#52b788",
    "Divergent (H9 earlier)": "#98e6c9",
    "Divergent (both later)": "#e63946",
    "Divergent (H1 later)": "#d90429",
    "Divergent (H9 later)": "#f07167",
    "Divergent (other)": "#b22222",
    "Intermediate": "#6c757d"
}

# =============================================================================
# A4-Optimized Visualization
# =============================================================================
plt.figure(figsize=(18, 12))

# Scale point size by the internal RT difference between human cell lines
size_values = df["hs_internal_diff_abs"].fillna(0)
max_size_val = size_values.max()
scaled_sizes = 30 + (size_values / max_size_val) * 150 if max_size_val > 0 else 50.0

# Plot each category separately to build legend handles
scatter_handles = {}
for category in sorted(df["category"].unique()):
    mask = df["category"] == category
    if mask.sum() > 0:
        handle = plt.scatter(
            df.loc[mask, "delta_H1"], df.loc[mask, "delta_H9"],
            s=scaled_sizes[mask], alpha=0.75,
            color=COLOR_MAP.get(category, "#777777"),
            edgecolor='white', linewidth=1.0
        )
        scatter_handles[category] = handle

ax = plt.gca()
# Plot Pf reference point (0,0) as a large star
pf_star_ref = ax.scatter(0, 0, s=400, color='black', marker='*', zorder=20)

# Add rectangles to highlight convergent and divergent regions
conv_rect = Rectangle(
    (-CONVERGENT_THRESHOLD, -CONVERGENT_THRESHOLD),
    2 * CONVERGENT_THRESHOLD, 2 * CONVERGENT_THRESHOLD,
    fill=False, linestyle='--', linewidth=3.0, ec='#219ebc', alpha=0.9
)
ax.add_patch(conv_rect)

div_rect = Rectangle(
    (-DIVERGENT_THRESHOLD, -DIVERGENT_THRESHOLD),
    2 * DIVERGENT_THRESHOLD, 2 * DIVERGENT_THRESHOLD,
    fill=False, linestyle='-', linewidth=2.5, ec='#e63946', alpha=0.5
)
ax.add_patch(div_rect)

# Add a diagonal line for reference (where H1_RT == H9_RT)
ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.4, linewidth=2.0)

# =============================================================================
# Legend and Final Touches
# =============================================================================
# Manually define legend order for clarity
legend_elements = [
    ("H1=H9 vs Pf", Line2D([0], [0], color='k', lw=2, linestyle='--')),
    ("Pf RT (reference)", pf_star_ref),
    (f"Convergent region (±{CONVERGENT_THRESHOLD})", Line2D([0], [0], color='#219ebc', lw=3.0, linestyle='--')),
    (f"Divergent threshold (±{DIVERGENT_THRESHOLD})", Line2D([0], [0], color='#e63946', lw=2.5, linestyle='-')),
    ("Intermediate", scatter_handles.get("Intermediate")),
    ("Convergent (H1)", scatter_handles.get("Convergent (H1)")),
    ("Convergent (H9)", scatter_handles.get("Convergent (H9)")),
    ("Convergent (both)", scatter_handles.get("Convergent (both)")),
    ("Divergent (H1 earlier)", scatter_handles.get("Divergent (H1 earlier)")),
    ("Divergent (H9 earlier)", scatter_handles.get("Divergent (H9 earlier)")),
    ("Divergent (both earlier)", scatter_handles.get("Divergent (both earlier)")),
    ("Divergent (H1 later)", scatter_handles.get("Divergent (H1 later)")),
    ("Divergent (H9 later)", scatter_handles.get("Divergent (H9 later)")),
    ("Divergent (both later)", scatter_handles.get("Divergent (both later)")),
]

# Filter out any missing handles (e.g., if a category has no data)
handles, labels = zip(*[(h, l) for l, h in legend_elements if h is not None])

plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=18)

# Set labels and title with increased font sizes for readability
plt.xlabel("Δ RT (Human H1 - P. falciparum)", fontsize=24)
plt.ylabel("Δ RT (Human H9 - P. falciparum)", fontsize=24)
plt.title("Conservation of Replication Timing between Human and P. falciparum", fontsize=28, pad=20)

# Increase font size of tick labels
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Symmetrize axes for better visual comparison
xlim, ylim = plt.xlim(), plt.ylim()
max_range = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]), 0.25)
plt.xlim(-max_range, max_range)
plt.ylim(-max_range, max_range)

plt.grid(True, linestyle=':', alpha=0.4, linewidth=1.5)
plt.tight_layout()

# Save the figure with high DPI for print quality
output_path = os.path.join(OUTPUT_DIR, "rt_pf_hs_conservation_scatter.pdf")
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.close()

print(f"A4-optimized plot saved to: {output_path}")


```






---

## rt_s01_coverage

```python


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict

# --- Configuration ---

# Simplified paths for a public repository.
# Assumes data is in a './data/' directory relative to the script.
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
PLOTS_DIR = Path("./plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input BED files
LEFT_FORKS = DATA_DIR / 'merged_36hpi_leftForks_DNAscent_forkSense.bed'
RIGHT_FORKS = DATA_DIR / 'merged_36hpi_rightForks_DNAscent_forkSense.bed'

# Analysis parameters
BIN_SIZES = [5000, 10000, 20000]  # in base pairs
RES_LABELS = {5000: '5kb', 10000: '10kb', 20000: '20kb'}

# Smoothing options
APPLY_SMOOTHING = True
SMOOTH_WINDOW_SIZE = 5  # Must be an odd number

# Plotting options
COLOR_COVERAGE = '#bdbdbd'
VISUALIZATION_TYPE = 'bars'  # Options: 'bars', 'heatmap', 'area'


def _read_bed_simple(path: Path) -> pd.DataFrame:
    """Reads a simple BED file (chrom, start, end)."""
    if not path.exists():
        raise FileNotFoundError(f"BED file not found: {path}")
    df = pd.read_csv(
        path, sep=r"\s+", header=None, comment='#', usecols=[0, 1, 2],
        names=['chrom', 'start', 'end'], dtype={'chrom': str}
    )
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna(subset=['start', 'end']).reset_index(drop=True)
    df[['start', 'end']] = df[['start', 'end']].astype(int)
    return df


def compute_coverage_vectorized(fork_data: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """
    Computes per-bin coverage and normalizes it by bin size.
    This makes coverage values comparable across different resolutions.
    """
    records = []
    chrom_lengths = fork_data.groupby('chrom')['end'].max().to_dict()

    for chrom, chrom_len in chrom_lengths.items():
        cforks = fork_data[fork_data['chrom'] == chrom]
        print(f"Processing Chr {chrom}: {len(cforks)} intervals, max position={chrom_len}")

        bin_edges = np.arange(0, int(chrom_len) + bin_size, bin_size)
        bin_starts = bin_edges[:-1].astype(int)
        num_bins = len(bin_starts)

        base_overlap = np.zeros(num_bins, dtype=float)
        left_counts = np.zeros(num_bins, dtype=int)
        right_counts = np.zeros(num_bins, dtype=int)

        if not cforks.empty:
            starts = cforks['start'].values.astype(int)
            ends = cforks['end'].values.astype(int)
            types = cforks['fork_type'].values

            # Clip intervals to chromosome bounds to prevent errors
            ends = np.minimum(ends, int(chrom_len))
            starts = np.maximum(starts, 0)
            valid_mask = ends > starts
            starts, ends, types = starts[valid_mask], ends[valid_mask], types[valid_mask]

            # Count fork start events per bin
            start_bins = np.minimum(np.maximum(starts // bin_size, 0), num_bins - 1)
            np.add.at(left_counts, start_bins[types == 'left'], 1)
            np.add.at(right_counts, start_bins[types == 'right'], 1)

            # Calculate total base-pair overlap for each bin
            for s, e in zip(starts, ends):
                sb = s // bin_size
                eb = (e - 1) // bin_size
                sb = int(max(0, min(sb, num_bins - 1)))
                eb = int(max(0, min(eb, num_bins - 1)))

                if sb == eb:
                    base_overlap[sb] += (e - s)
                else:
                    base_overlap[sb] += ((sb + 1) * bin_size - s) # Partial first bin
                    base_overlap[eb] += (e - eb * bin_size)      # Partial last bin
                    if eb - sb > 1:
                        base_overlap[sb+1:eb] += bin_size        # Full bins in between

        # Normalize by bin size to get average per-base coverage (depth)
        coverage = base_overlap / float(bin_size)

        df = pd.DataFrame({
            'chrom': [chrom] * num_bins, 'start': bin_starts,
            'end': bin_edges[1:].astype(int), 'mid': ((bin_starts + bin_edges[1:]) // 2).astype(int),
            'left_counts': left_counts, 'right_counts': right_counts,
            'coverage': coverage
        })
        records.append(df)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def smooth_coverage(coverage_df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    """Applies a centered rolling mean to the coverage data."""
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd for centering

    smoothed_df = coverage_df.copy()
    for chrom in coverage_df['chrom'].unique():
        chr_mask = coverage_df['chrom'] == chrom
        smoothed_coverage = coverage_df.loc[chr_mask, 'coverage'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        smoothed_df.loc[chr_mask, 'coverage'] = smoothed_coverage.values

    return smoothed_df


def plot_coverage(ax, chr_df: pd.DataFrame):
    """Generic plotting function to draw coverage on a given axis."""
    if VISUALIZATION_TYPE == 'bars':
        mask = chr_df['coverage'] > 0
        if mask.any():
            x_mb = chr_df.loc[mask, 'start'].values / 1e6
            widths = (chr_df.loc[mask, 'end'].values - chr_df.loc[mask, 'start'].values) / 1e6
            vals = chr_df.loc[mask, 'coverage'].values
            colors = plt.cm.Greys(0.3 + 0.6 * (vals / vals.max())) # Grey gradient
            ax.bar(x_mb, vals, width=widths, color=colors, align='edge', alpha=0.8)
        ax.set_ylabel('Coverage')
        max_val = float(chr_df['coverage'].max()) if not chr_df.empty else 1.0
        ax.set_ylim(0, max_val * 1.05 if max_val > 0 else 1.0)

    elif VISUALIZATION_TYPE == 'heatmap':
        x_mb = chr_df['start'].values / 1e6
        widths = (chr_df['end'].values - chr_df['start'].values) / 1e6
        vals = chr_df['coverage'].values
        max_val = vals.max() if len(vals) > 0 else 1.0
        for x, w, cov in zip(x_mb, widths, vals):
            if cov > 0:
                intensity = min(cov / max_val, 1.0)
                color = plt.cm.Greys(0.2 + 0.7 * intensity)
                rect = plt.Rectangle((x, 0), w, 1, facecolor=color, edgecolor='none')
                ax.add_patch(rect)
        ax.set_ylabel('Coverage Intensity')
        ax.set_ylim(0, 1)

    elif VISUALIZATION_TYPE == 'area':
        x_mb = chr_df['mid'].values / 1e6
        vals = chr_df['coverage'].values
        ax.fill_between(x_mb, vals, alpha=0.6, color=COLOR_COVERAGE)
        ax.plot(x_mb, vals, color='black', linewidth=0.5, alpha=0.8)
        ax.set_ylabel('Coverage')
        max_val = float(chr_df['coverage'].max()) if not chr_df.empty else 1.0
        ax.set_ylim(0, max_val * 1.05 if max_val > 0 else 1.0)


def generate_single_resolution_pdf(coverage_df: pd.DataFrame, label: str, pdf_out: Path):
    """Creates a multi-page PDF with one chromosome per page for a single resolution."""
    chroms = sorted(coverage_df['chrom'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    with PdfPages(pdf_out) as pdf:
        for chrom in chroms:
            chr_df = coverage_df[coverage_df['chrom'] == chrom].sort_values('mid')
            if chr_df.empty: continue

            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.suptitle(f"Chromosome {chrom} — Coverage ({label})", fontsize=10)
            plot_coverage(ax, chr_df)
            ax.set_xlabel('Genomic Position (Mb)', labelpad=10)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved PDF: {pdf_out}")


def generate_stacked_resolution_pdf(chrom: str, dfs_by_resolution: Dict[int, pd.DataFrame], out_path: Path):
    """Creates a single-page PDF for one chromosome, stacking all resolution panels."""
    fig, axes = plt.subplots(len(BIN_SIZES), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Chromosome {chrom} — Coverage by Resolution", fontsize=12)

    for ax, bin_size in zip(axes, BIN_SIZES):
        df = dfs_by_resolution.get(bin_size)
        if df is None or df[df['chrom'] == chrom].empty:
            ax.set_visible(False)
            continue
        
        chr_df = df[df['chrom'] == chrom].sort_values('mid')
        plot_coverage(ax, chr_df)
        ax.set_title(f"{RES_LABELS[bin_size]} bins", loc='right', fontsize=9)

    axes[-1].set_xlabel('Genomic Position (Mb)', labelpad=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved stacked PDF: {out_path}")


def main():
    """Main script execution."""
    left = _read_bed_simple(LEFT_FORKS)
    right = _read_bed_simple(RIGHT_FORKS)
    left['fork_type'], right['fork_type'] = 'left', 'right'
    forks = pd.concat([left, right], ignore_index=True)
    
    chroms = sorted(forks['chrom'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    print(f"Found chromosomes: {chroms}")

    dfs_by_resolution = {}
    for b in BIN_SIZES:
        label = RES_LABELS[b]
        print(f"\n--- Processing resolution: {label} ---")
        coverage_df = compute_coverage_vectorized(forks, b)

        if APPLY_SMOOTHING:
            print(f"Applying smoothing with window size {SMOOTH_WINDOW_SIZE}")
            coverage_df = smooth_coverage(coverage_df, SMOOTH_WINDOW_SIZE)

        out_tbl = RESULTS_DIR / f"rt_coverage_{label}.txt"
        coverage_df.to_csv(out_tbl, sep='\t', index=False)
        print(f"Saved table: {out_tbl}")

        pdf_out = PLOTS_DIR / f"rt_coverage_{label}.pdf"
        generate_single_resolution_pdf(coverage_df, label, pdf_out)
        dfs_by_resolution[b] = coverage_df

    print("\n--- Generating stacked resolution plots per chromosome ---")
    for chrom in chroms:
        chrom_pdf = PLOTS_DIR / f"rt_{chrom}_coverage_stacked.pdf"
        generate_stacked_resolution_pdf(chrom, dfs_by_resolution, chrom_pdf)

    print("\n--- Merging all resolution tables ---")
    merged_df = None
    for b, df in dfs_by_resolution.items():
        suffix = '_' + RES_LABELS[b]
        df_renamed = df[['chrom', 'start', 'mid', 'coverage', 'left_counts', 'right_counts']].rename(
            columns=lambda c: c + suffix if c not in ['chrom', 'start', 'mid'] else c
        )
        if merged_df is None:
            merged_df = df_renamed
        else:
            merged_df = pd.merge(merged_df, df_renamed, how='outer', on=['chrom', 'start', 'mid'])
    
    merged_out = RESULTS_DIR / 'rt_coverage_all_resolutions.txt'
    merged_df.to_csv(merged_out, sep='\t', index=False)
    print(f"Saved merged table: {merged_out}")

    print('\nAnalysis complete. Outputs are in:')
    print(f'-> Results: {RESULTS_DIR.resolve()}')
    print(f'-> Plots:   {PLOTS_DIR.resolve()}')


if __name__ == '__main__':
    main()


```






---

## art_01_preproceccing

```shell

# 01: convert fast5 to pod5
INPUT_DIR="/home/xh368/rds/rds-huang_xr-CGClDViiOBk/ONT_data/2023_07_28_LS_ONT_Pfal_DHA_R10/fast5/20230725_0319_MN28426_FAX16930_416af3c4/fast5"
OUTPUT_DIR="/home/xh368/rds/rds-huang_xr-CGClDViiOBk/ONT_data/2023_07_28_LS_ONT_Pfal_DHA_R10/fast5/20230725_0319_MN28426_FAX16930_416af3c4/pod5"

source /home/xh368/anaconda3/bin/activate
conda activate /home/xh368/anaconda3/envs/pod5/
mkdir -p "$OUTPUT_DIR"

pod5 convert fast5 "$INPUT_DIR" --output "$OUTPUT_DIR" --recursive --threads 32


# 02: basecalling
DORADO="/software/dorado-1.0.1-linux-x64/bin/dorado basecaller"
INPUT="/ont_data/pod5/"
GENOME="/genomes/falciparum/GCF_000002765.6_GCA_000002765_genomic.fna"
CONFIG="/dorado-1.0.1-linux-x64/models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0"
OUTPUT="/ont_data/basecall"
srun $DORADO -x cuda:0 -r --reference "$GENOME" --output-dir "$OUTPUT" --mm2-opts "-x map-ont" "$CONFIG" "$INPUT"


# 03: demultiplexing
DORADO="/software/dorado-1.0.1-linux-x64/bin/dorado demux"
INPUT="/ont_data/basecall/calls.bam"
OUTPUT="/ont_data/demux/"
srun ${DORADO} --kit-name SQK-NBD114-24 --output-dir "${OUTPUT}" "${INPUT}" --threads 32 --no-trim


# 04: index
/software/DNAscent.sif index \
  -f /ont_data/pod5 \
  -o /ont_data/index/index.dnascent


# 05: alignment
export HDF5_PLUGIN_PATH=/software/ont-vbz-hdf-plugin-1.0.1-Linux/usr/local/hdf5/lib/plugin/
DNASCENT="/software/DNAscent/bin/DNAscent"
EXECUTABLE="align"
DEMUX="/ont_data/demux"
REFERENCE="/genomes/falciparum/GCF_000002765.6_GCA_000002765_genomic.fna"
INDEX="/ont_data/index/index.dnascent"
BAM=$(ls ${DEMUX_DIR}/*_SQK-NBD114-24_barcode0${SLURM_ARRAY_TASK_ID}.bam 2>/dev/null)
OUTPUT="/ont_data/align/barcode0${SLURM_ARRAY_TASK_ID}_DNAscent_v4.0.3.align"
srun $DNASCENT $EXECUTABLE -b "$BAM" -r "$REFERENCE" -i "$INDEX" -o "$OUTPUT" -t 32 -q 20 -l 2000 -m 10000
exit_code=$?


```


---



## art_02_9mer

```python


import os
from collections import Counter
from pathlib import Path

def process_and_filter_9mers(file_paths, output_file):
    """
    Reads .align files, counts 9-mers per barcode, applies filters,
    and writes the results to a single output file.
    """
    barcode_9mer_counts = {}

    # --- Step 1: Count 9-mers from all input files ---
    print("Starting to count 9-mers from input files...")
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        
        try:
            # Extract barcode from filename (e.g., 'barcode01')
            base_name = os.path.basename(file_path)
            barcode = base_name.split('_')[0]

            if barcode not in barcode_9mer_counts:
                barcode_9mer_counts[barcode] = Counter()

            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    columns = line.strip().split()
                    if len(columns) > 1:
                        nine_mer = columns[1]
                        if len(nine_mer) == 9:
                            barcode_9mer_counts[barcode][nine_mer] += 1
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    print("Counting complete.")

    # --- Step 2: Apply filters and write to output file ---
    print(f"Applying filters and writing results to {output_file}...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as out_f:
        out_f.write("Barcode\t9mer\tCount\n")  # Write header
        
        for barcode, nine_mer_counts in barcode_9mer_counts.items():
            for nine_mer, count in nine_mer_counts.items():
                
                # --- Filtering Logic ---
                middle_5mer = nine_mer[2:7]
                
                # Condition 1: Middle 5-mer has >= 3 'A's OR the 9-mer has no 'A's.
                cond1 = (middle_5mer.count("A") >= 3) or (nine_mer.count("A") == 0)
                
                # Condition 2: The 9-mer is composed of at least 3 unique bases.
                cond2 = len(set(nine_mer)) >= 3
                
                if cond1 and cond2:
                    out_f.write(f"{barcode}\t{nine_mer}\t{count}\n")
    
    print("Processing complete!")

# --- Configuration ---

# Assume a project structure like:
# /project_root
#   /data
#     - barcode01_DNAscent_v4.0.3.align
#     - ...
#   /results
#   - this_script.py

# Generate input file paths automatically
# This is more flexible than a hardcoded list.
input_dir = "data"
barcodes = [f"barcode{i:02d}" for i in range(1, 8)]
input_files = [
    os.path.join(input_dir, f"{bc}_DNAscent_v4.0.3.align") for bc in barcodes
]

# Define the simplified output file path
output_file = "results/filtered_9mer_list.txt"

# --- Execute Script ---
if __name__ == "__main__":
    process_and_filter_9mers(input_files, output_file)



```






---

## art_03_gmm

```python


import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Set backend for running on servers without a display
matplotlib.use('Agg')

# =============================================================================
# Configuration
# =============================================================================
# Simplified paths for public code sharing.
# Assumes a directory structure like this:
# ./
# |- data/
# |  |- barcodes/
# |     |- barcode01.align
# |     |- ...
# |- results/
# |  |- filtered_signal_9mer_list.txt
# |- plots/
# |- models/
# |  |- r10.4.1_400bps.nucleotide.9mer.model
# |- your_script.py

RESULTS_DIR = "results"
PLOT_DIR = "plots/9mer_distributions"
DATA_DIR = "data/barcodes"
MODEL_DIR = "models"

FILTERED_9MER_FILE = os.path.join(RESULTS_DIR, "filtered_signal_9mer_list.txt")
ONT_MODEL_FILE = os.path.join(MODEL_DIR, "r10.4.1_400bps.nucleotide.9mer.model")

os.makedirs(PLOT_DIR, exist_ok=True)

# =============================================================================
# Load and Select Top 9-mers
# =============================================================================
try:
    def has_diverse_center(mer):
        """Check if the central 5-mer has at least 3 unique bases."""
        return len(set(mer[2:7])) >= 3

    # Load and filter the 9-mer data to select the top 200 most abundant ones
    # that meet quality criteria (present in all 7 samples, count >= 200, diverse center).
    filtered_9mer_df = pd.read_csv(FILTERED_9MER_FILE, sep='\t')
    filtered_9mer_df['count'] = pd.to_numeric(filtered_9mer_df['count'], errors='coerce')

    filtered_9mer_summary = (
        filtered_9mer_df.groupby('9mer')
        .filter(lambda x: len(x) == 7 and all(x['count'] >= 200) and has_diverse_center(x.name))
        .groupby('9mer')['count']
        .sum()
        .nlargest(200)
    )
    selected_9mers = filtered_9mer_summary.index.tolist()
    print(f"Selected {len(selected_9mers)} 9-mers for plotting.")

except Exception as e:
    logging.error(f"Error loading or processing 9-mer file: {e}")
    selected_9mers = []

# =============================================================================
# Helper Functions
# =============================================================================
def reverse_complement(sequence):
    """Compute the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(sequence))

def load_9mer_data(filepath, target_9mer):
    """Load signal data for a specific 9-mer from a DNAscent alignment file."""
    data = []
    strand = "fwd"
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # The header line contains strand information
                    strand = line.strip().split(' ')[-1]
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) > 2:
                    sequence = parts[1]
                    if strand == "rev":
                        sequence = reverse_complement(sequence)
                    
                    if sequence == target_9mer:
                        try:
                            data.append(float(parts[2]))
                        except ValueError:
                            pass # Ignore lines with invalid signal values
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    return data

def load_ont_model(filepath, target_9mer):
    """Load the expected signal mean and std dev for a 9-mer from an ONT model file."""
    mean_signal = None
    std_signal = 0.14  # Default standard deviation for this ONT model
    try:
        with open(filepath, 'r') as f:
            for line in f:
                kmer, mean_str = line.strip().split('\t')
                if kmer == target_9mer:
                    mean_signal = float(mean_str)
                    break
    except FileNotFoundError:
        logging.error(f"ONT model file not found: {filepath}")
    return mean_signal, std_signal

# =============================================================================
# Plotting
# =============================================================================
# Define input files and their corresponding labels for the plots
filepaths = [os.path.join(DATA_DIR, f'barcode0{i}_DNAscent_v4.0.3.align') for i in range(1, 8)]
labels = [
    'DMSO Control',
    'DHA + Heme (37°C)',
    'DHA + Heme (70°C)',
    'DHA + FeSO₄ (37°C)',
    'DHA + FeCl₃ (37°C)',
    'DMSO Control (In Vivo)',
    'DHA (In Vivo)'
]

# Generate a plot for each selected 9-mer
for target_9mer in selected_9mers:
    ont_mean, ont_std = load_ont_model(ONT_MODEL_FILE, target_9mer)
    if ont_mean is None:
        logging.warning(f"Missing ONT model data for {target_9mer}. Skipping.")
        continue

    # Simulate ONT model data for plotting its distribution
    ont_model_data = np.random.normal(loc=ont_mean, scale=ont_std, size=1000)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    y_max_values = []

    for i, (filepath, label) in enumerate(zip(filepaths, labels)):
        ax = axes[i]
        data = load_9mer_data(filepath, target_9mer)

        if not data:
            ax.text(0.5, 0.5, f"No data for\n{label}", ha='center', va='center', transform=ax.transAxes)
        else:
            # Plot the kernel density estimate of the observed data
            sns.kdeplot(data, color="#F5D47D", linewidth=2, ax=ax, label="Observed")
            
            # Fit a Gaussian Mixture Model to find potential sub-populations
            gmm = GaussianMixture(n_components=2, random_state=42).fit(np.array(data).reshape(-1, 1))
            
            x_fit = np.linspace(min(data) - 0.5, max(data) + 0.5, 1000)
            
            # Plot the individual Gaussian components from the GMM
            for mean, cov, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
                y_fit = weight * norm.pdf(x_fit, loc=mean[0], scale=np.sqrt(cov[0][0]))
                ax.plot(x_fit, y_fit, linestyle='--')

            # Plot the expected distribution from the ONT model
            sns.kdeplot(ont_model_data, color="#AECDE0", linewidth=2, ax=ax, label="ONT Model")
            
            y_max_values.append(ax.get_ylim()[1])
            if i == 0: ax.legend()

        ax.set_title(label, fontsize=10)

    # Set a consistent y-axis limit across all subplots
    if y_max_values:
        global_ymax = max(y_max_values) * 1.05
        for ax in axes[:len(filepaths)]:
            ax.set_ylim(top=global_ymax)

    # Remove unused subplots
    for i in range(len(filepaths), len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f"Signal Distribution for 9-mer: {target_9mer}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = os.path.join(PLOT_DIR, f'9mer_{target_9mer}.png')
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_file}")


```






---


## art_04_model_visualisation

```shell

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Configure Matplotlib for consistent PDF output
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = ['Arial']

# --- Configuration ---
# Assumes the script is run from the project root.
# Data is expected in './data/roc_data_barcode_XX' directories.
DATA_DIR = Path("./data")
PLOTS_DIR = Path("./plots")

# Input directories for ROC data
ROC_DIR_A = DATA_DIR / 'roc_data_barcode12'
ROC_DIR_B = DATA_DIR / 'roc_data_barcode67'

# Output file
OUTPUT_PDF = PLOTS_DIR / 'roc_comparison_plot.pdf'

# Ensure the output directory exists
PLOTS_DIR.mkdir(exist_ok=True)


def load_roc_data(roc_dir: Path):
    """Loads all roc_data_epochXX.npz files from a directory."""
    roc_data_list = []
    # Patterns to match different possible filenames
    patterns = [re.compile(r'roc_data_epoch(\d+)\.npz'), re.compile(r'roc_epoch(\d+)\.npz')]

    if not roc_dir.exists():
        print(f"Warning: Directory not found: {roc_dir}")
        return []

    for filename in os.listdir(roc_dir):
        for pattern in patterns:
            match = pattern.match(filename)
            if match:
                epoch_num = int(match.group(1))
                file_path = roc_dir / filename
                try:
                    data = np.load(file_path)
                    roc_data_list.append({
                        'epoch': epoch_num,
                        'fpr': data['fpr'],
                        'tpr': data['tpr'],
                        'auc': float(data['auc'])
                    })
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
                break  # Matched one pattern, move to the next file

    # Sort the data by epoch number to ensure correct color progression
    roc_data_list.sort(key=lambda x: x['epoch'])
    return roc_data_list

def plot_single_panel(ax, roc_data, title):
    """Plots a single ROC curve panel with a color gradient for epochs."""
    if not roc_data:
        ax.text(0.5, 0.5, 'No data found', ha='center', va='center', color='gray')
        ax.set_title(title, fontsize=18, pad=15)
        return

    num_epochs = len(roc_data)
    # A blue-to-orange color gradient for the epochs
    colors = LinearSegmentedColormap.from_list(
        "custom_gradient", ['#8EC5FC', '#FF9A8B']
    )(np.linspace(0, 1, num_epochs))
    
    best_epoch_color = '#d62728' # A standout red for the best epoch

    best_epoch_idx = np.argmax([d['auc'] for d in roc_data])

    # Plot all epochs, highlighting the one with the highest AUC
    for i, data in enumerate(roc_data):
        is_best = (i == best_epoch_idx)
        ax.plot(
            data['fpr'], data['tpr'],
            color=best_epoch_color if is_best else colors[i],
            lw=3.5 if is_best else 1.5,
            alpha=1.0 if is_best else 0.4,
            zorder=10 if is_best else 2,
            label=f"Best (Epoch {data['epoch']}, AUC = {data['auc']:.4f})" if is_best else None
        )

    ax.plot([0, 1], [0, 1], color='dimgray', lw=1.5, linestyle='--', label='Random Guess (AUC=0.5)', zorder=1)

    # Aesthetics and labels
    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', framealpha=0.85, shadow=True)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def plot_two_panel_roc(roc_data_a, roc_data_b, output_path):
    """Creates and saves a two-panel ROC curve figure."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5))
    fig.suptitle('Model Performance Across Training Epochs', fontsize=22, fontweight='bold', y=0.98)

    plot_single_panel(ax1, roc_data_a, r'$\it{in\ vitro}$: Barcode01 (DMSO) vs Barcode02 (DHA)')
    plot_single_panel(ax2, roc_data_b, r'$\it{in\ vivo}$: Barcode06 (DMSO) vs Barcode07 (DHA)')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved plot to: {output_path}")

if __name__ == "__main__":
    print("Generating two-panel ROC curve comparison plot...")

    print("\nLoading data for Panel A...")
    roc_data_a = load_roc_data(ROC_DIR_A)

    print("\nLoading data for Panel B...")
    roc_data_b = load_roc_data(ROC_DIR_B)

    if roc_data_a or roc_data_b:
        plot_two_panel_roc(roc_data_a, roc_data_b, OUTPUT_PDF)
    else:
        print("\nCould not generate plot: No valid ROC data was found.")

    print("\nOperation complete.")




```




