"""
Step 1B: Cell-Type Deconvolution

Implements:
  - Synthetic Mathys 2019 reference generation
  - NNLS deconvolution
  - Cell-type proportion estimation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import scanpy as sc
import warnings
import os
import logging

warnings.filterwarnings('ignore')


def generate_synthetic_mathys_reference(raw_data_dir, test_mode=False):
    """Generate synthetic snRNA-seq reference."""
    np.random.seed(42)

    n_cells = 1000 if test_mode else 8066
    n_genes = 500 if test_mode else 10000
    n_subjects = 10 if test_mode else 48

    cell_types = ['Ex', 'In', 'Ast', 'Oli', 'Mic', 'OPCs']
    cell_type_proportions = [0.35, 0.15, 0.20, 0.15, 0.10, 0.05]

    cell_type_labels = np.random.choice(cell_types, size=n_cells, p=cell_type_proportions)
    subject_ids = np.random.choice([f'ROSMAP_{i:03d}' for i in range(n_subjects)], size=n_cells)
    gene_names = [f'GENE_{i}' for i in range(n_genes)]

    X_counts = np.zeros((n_cells, n_genes))
    for ct in cell_types:
        mask = cell_type_labels == ct
        n_ct_cells = mask.sum()
        ct_baseline = np.random.exponential(scale=0.5, size=n_genes)
        ct_noise = np.random.normal(0, 1, size=(n_ct_cells, n_genes))
        X_counts[mask, :] = np.maximum(ct_baseline[np.newaxis, :] + ct_noise, 0)

    adata = sc.AnnData(
        X=X_counts,
        obs=pd.DataFrame({
            'cell_type': cell_type_labels,
            'broad.cell.type': cell_type_labels,
            'subject_id': subject_ids
        }, index=[f'cell_{i}' for i in range(n_cells)]),
        var=pd.DataFrame({'gene_id': gene_names}, index=gene_names)
    )

    os.makedirs(raw_data_dir, exist_ok=True)
    adata_file = f'{raw_data_dir}/mathys_reference.h5ad'
    adata.write(adata_file)

    logging.info(f"  Generated synthetic reference: {adata.shape}")
    return adata_file


def nnls_deconvolve(bulk_profile, reference_matrix):
    """Non-negative least squares deconvolution.

    Args:
        bulk_profile: 1D array of protein abundances (n_proteins,)
        reference_matrix: 2D array of cell-type profiles (n_cell_types, n_proteins)

    Returns:
        proportions: 1D array of cell-type proportions summing to 1
    """
    from scipy.optimize import nnls

    # NNLS solves min ||Ax - b||_2 where A is (m,n), b is (m,)
    # We have reference_matrix (6, n_proteins) and bulk_profile (n_proteins,)
    # We need to solve for proportions that minimize ||A^T x - b||
    # So A needs to be (n_proteins, n_cell_types)
    A = reference_matrix.T  # Transpose to (n_proteins, n_cell_types)
    b = bulk_profile

    # Solve NNLS
    proportions, _ = nnls(A, b)

    # Normalize to sum to 1
    proportions = np.maximum(proportions, 0)
    if proportions.sum() > 0:
        proportions = proportions / proportions.sum()
    else:
        proportions = np.ones_like(proportions) / len(proportions)

    return proportions


def save_deconvolved_data(proportions_df, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    props_file = f"{processed_dir}/cell_type_proportions.csv"
    proportions_df.to_csv(props_file)
    logging.info(f"  Saved: cell_type_proportions.csv ({proportions_df.shape})")
    return props_file


def plot_proportions(proportions_df, metadata_df, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    cell_types = proportions_df.columns
    x = np.arange(len(proportions_df))
    bottom = np.zeros(len(proportions_df))

    colors = plt.cm.Set3(np.linspace(0, 1, len(cell_types)))

    for i, ct in enumerate(cell_types):
        ax.bar(x, proportions_df[ct], bottom=bottom, label=ct, color=colors[i], alpha=0.8)
        bottom += proportions_df[ct].values

    ax.set_ylabel('Cell Type Proportion', fontsize=12)
    ax.set_title('Cell-Type Proportions by Patient', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    props_file = f"{results_dir}/cell_type_proportions.png"
    plt.savefig(props_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: cell_type_proportions.png")

    return proportions_df


def main(data_dir="data", results_dir="results", skip_deconvolution=False, test_mode=False):
    """
    Run Step 1B: Cell-type deconvolution.

    Returns:
        dict: Results with keys: n_cell_types, status
    """
    logger = logging.getLogger("Step1B")

    try:
        raw_data_dir = f"{data_dir}/raw"
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        logger.info("="*70)
        logger.info("STEP 1B: Cell-Type Deconvolution")
        logger.info("="*70)

        if skip_deconvolution:
            logger.info("  [SKIP] Deconvolution skipped per user request")
            logger.info("  Creating placeholder cell_type_proportions.csv...")

            # Load proteomics to get patient IDs
            proteomics_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
            if not os.path.exists(proteomics_file):
                raise FileNotFoundError(f"Cannot find {proteomics_file}")

            proteomics_df = pd.read_csv(proteomics_file, index_col=0)
            cell_types = ['ct_Ex', 'ct_In', 'ct_Ast', 'ct_Oli', 'ct_Mic', 'ct_OPCs']

            # Create equal proportions
            n_patients = len(proteomics_df)
            proportions_df = pd.DataFrame(
                np.ones((n_patients, 6)) / 6,
                index=proteomics_df.index,
                columns=cell_types
            )

            proportions_file = save_deconvolved_data(proportions_df, processed_dir)
            logger.info("  Deconvolution skipped successfully")

            return {
                'n_cell_types': 6,
                'n_patients': n_patients,
                'status': 'SKIPPED'
            }

        # Load or generate reference
        logger.info("[1/5] Loading reference...")
        ref_file = f"{raw_data_dir}/mathys_reference.h5ad"
        if not os.path.exists(ref_file):
            logger.info("  Reference not found, generating synthetic...")
            generate_synthetic_mathys_reference(raw_data_dir, test_mode=test_mode)

        adata = sc.read_h5ad(ref_file)
        logger.info(f"  Loaded reference: {adata.shape}")

        # Load bulk
        logger.info("[2/5] Loading bulk proteomics...")
        proteomics_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
        proteomics_df = pd.read_csv(proteomics_file, index_col=0)
        logger.info(f"  Loaded bulk: {proteomics_df.shape}")

        # Compute reference profiles
        logger.info("[3/5] Computing reference profiles...")
        cell_types = adata.obs['cell_type'].unique()
        reference_matrix = np.zeros((len(cell_types), proteomics_df.shape[1]))

        for i, ct in enumerate(sorted(cell_types)):
            ct_mask = adata.obs['cell_type'] == ct
            reference_matrix[i, :min(adata.shape[1], proteomics_df.shape[1])] = \
                adata.X[ct_mask, :proteomics_df.shape[1]].mean(axis=0)

        # Deconvolve
        logger.info("[4/5] Running NNLS deconvolution...")
        proportions_list = []
        for patient in proteomics_df.index:
            bulk_profile = proteomics_df.loc[patient].values
            proportions = nnls_deconvolve(bulk_profile, reference_matrix)
            proportions_list.append(proportions)

        ct_names = [f'ct_{ct}' for ct in sorted(cell_types)]
        proportions_df = pd.DataFrame(
            proportions_list,
            index=proteomics_df.index,
            columns=ct_names
        )

        logger.info(f"  Deconvolved {len(proteomics_df)} patients")

        # Save
        logger.info("[5/5] Saving outputs...")
        proportions_file = save_deconvolved_data(proportions_df, processed_dir)

        # Plot
        metadata_file = f"{processed_dir}/rosmap_metadata.csv"
        if os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file, index_col=0)
            plot_proportions(proportions_df, metadata_df, results_1_dir)

        logger.info("="*70)
        logger.info("STEP 1B COMPLETE")
        logger.info("="*70)

        return {
            'n_cell_types': len(cell_types),
            'n_patients': len(proportions_df),
            'status': 'PASS'
        }

    except Exception as e:
        logger.error(f"Step 1B failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
