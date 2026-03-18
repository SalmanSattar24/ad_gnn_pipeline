"""
Step 1C: Disease Pseudotime Computation

Implements:
  - Feature selection (top variable proteins)
  - PCA dimensionality reduction
  - UMAP embedding
  - Diffusion pseudotime (DPT) with clinical validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import scanpy as sc
import warnings
import os
import logging

warnings.filterwarnings('ignore')


def load_merged_data(processed_dir):
    """Load proteomics, metadata, and cell-type proportions."""
    proteomics_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
    metadata_file = f"{processed_dir}/rosmap_metadata.csv"
    proportions_file = f"{processed_dir}/cell_type_proportions.csv"

    proteomics_df = pd.read_csv(proteomics_file, index_col=0)
    metadata_df = pd.read_csv(metadata_file, index_col=0)

    proportions_df = None
    if os.path.exists(proportions_file):
        proportions_df = pd.read_csv(proportions_file, index_col=0)

    logging.info(f"  Loaded proteomics: {proteomics_df.shape}")
    logging.info(f"  Loaded metadata: {metadata_df.shape}")
    if proportions_df is not None:
        logging.info(f"  Loaded proportions: {proportions_df.shape}")

    return proteomics_df, metadata_df, proportions_df


def select_top_variable_proteins(proteomics_df, n_top=500):
    """Select top variable proteins."""
    variances = proteomics_df.var(axis=0)
    top_proteins = variances.nlargest(n_top).index.tolist()
    proteomics_subset = proteomics_df[top_proteins]
    logging.info(f"  Selected {len(top_proteins)} most variable proteins")
    return proteomics_subset


def compute_pca(proteomics_df, n_components=50):
    """Compute PCA."""
    pca = PCA(n_components=min(n_components, proteomics_df.shape[0]-1))
    pca_coords = pca.fit_transform(proteomics_df)
    explained_var = pca.explained_variance_ratio_.sum()
    logging.info(f"  PCA: {pca.n_components_} components, {explained_var*100:.1f}% variance explained")
    return pca_coords, pca


def compute_umap(pca_coords, n_neighbors=15, min_dist=0.3):
    """Compute UMAP from PCA coordinates."""
    adata = sc.AnnData(X=pca_coords)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist)
    umap_coords = adata.obsm['X_umap']
    logging.info(f"  UMAP computed")
    return umap_coords, adata


def select_dpt_root(adata, metadata_df):
    """Select root cell for DPT based on cognitive status and Braak stage."""
    # Find cognitively normal with low Braak
    candidates = []

    for idx, row in metadata_df.iterrows():
        cogdx_variants = ['cogdx', 'COGDX']
        braak_variants = ['braaksc', 'BRAAKSC', 'braak']

        cogdx_val = None
        braak_val = None

        for var in cogdx_variants:
            if var in metadata_df.columns:
                cogdx_val = row[var]
                break

        for var in braak_variants:
            if var in metadata_df.columns:
                braak_val = row[var]
                break

        if cogdx_val == 1 and braak_val is not None and braak_val <= 1:
            candidates.append(idx)

    if candidates:
        # Choose the one closest to center of UMAP space
        if hasattr(adata, 'obsm') and 'X_umap' in adata.obsm:
            umap_center = adata.obsm['X_umap'].mean(axis=0)
            distances = {}
            for cand in candidates:
                cand_idx = list(adata.obs_names).index(cand) if cand in adata.obs_names else None
                if cand_idx is not None:
                    dist = np.linalg.norm(adata.obsm['X_umap'][cand_idx] - umap_center)
                    distances[cand] = dist
            if distances:
                root = min(distances, key=distances.get)
            else:
                root = candidates[0]
        else:
            root = candidates[0]
    else:
        # Fallback: use first sample
        root = adata.obs_names[0]

    logging.info(f"  Selected root cell: {root}")
    return root


def compute_pseudotime(adata, root, n_dcs=10):
    """Compute diffusion pseudotime."""
    sc.tl.diffmap(adata, n_comps=n_dcs)

    # Convert root cell name to obs index if needed
    if root in adata.obs_names:
        root_idx = list(adata.obs_names).index(root)
        adata.uns['iroot'] = root_idx

    sc.tl.dpt(adata)
    pseudotime = adata.obs['dpt_pseudotime'].values
    logging.info(f"  Pseudotime range: {pseudotime.min():.3f} - {pseudotime.max():.3f}")
    return pseudotime


def validate_pseudotime(pseudotime, metadata_df, logger):
    """Validate pseudotime against clinical measures."""
    validations = {
        'mmse': ('mmse', -1, -0.30),
        'braak': ('braaksc', 1, 0.30),
        'cerad': ('ceradsc', -1, -0.30),
        'cogdx': ('cogdx', 1, 0.30)
    }

    valid_count = 0
    for key, (col, sign, threshold) in validations.items():
        col_variants = [col, col.upper()]
        found = False

        for var in col_variants:
            if var in metadata_df.columns:
                clinical_vals = metadata_df[var].values
                if np.isnan(clinical_vals).sum() < len(clinical_vals) * 0.5:
                    # Compute correlation
                    mask = ~(np.isnan(pseudotime) | np.isnan(clinical_vals))
                    if mask.sum() > 3:
                        rho, pval = spearmanr(pseudotime[mask], clinical_vals[mask])
                        if sign * rho > threshold:
                            valid_count += 1
                            logger.info(f"    {key}: rho={rho:.3f} (p={pval:.2e})")
                        found = True
                        break

    logger.info(f"  Validation: {valid_count}/4 measures passed")
    return valid_count >= 4  # At least 4 measures should correlate


def save_pseudotime_data(pseudotime, adata, metadata_df, proportions_df, processed_dir):
    """Save pseudotime scores and master table."""
    os.makedirs(processed_dir, exist_ok=True)

    # Pseudotime file
    pseudotime_df = pd.DataFrame({
        'dpt_pseudotime': pseudotime,
        'umap_1': adata.obsm['X_umap'][:, 0],
        'umap_2': adata.obsm['X_umap'][:, 1]
    }, index=adata.obs_names)

    pseudo_file = f"{processed_dir}/pseudotime_scores.csv"
    pseudotime_df.to_csv(pseudo_file)
    logging.info(f"  Saved: pseudotime_scores.csv")

    # Master table
    master_df = metadata_df.copy()
    master_df['dpt_pseudotime'] = pseudotime_df['dpt_pseudotime']
    master_df['umap_1'] = pseudotime_df['umap_1']
    master_df['umap_2'] = pseudotime_df['umap_2']

    if proportions_df is not None:
        for col in proportions_df.columns:
            master_df[col] = proportions_df[col]

    master_file = f"{processed_dir}/master_patient_table.csv"
    master_df.to_csv(master_file)
    logging.info(f"  Saved: master_patient_table.csv ({master_df.shape})")

    return master_file


def plot_pseudotime_umap(adata, metadata_df, results_dir):
    """Plot 4-panel UMAP colored by different metrics."""
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Diagnosis
    ax = axes[0, 0]
    for diag in metadata_df.get('diagnosis', {}).unique() if 'diagnosis' in metadata_df.columns else []:
        mask = metadata_df['diagnosis'] == diag
        ax.scatter(adata.obsm['X_umap'][mask, 0], adata.obsm['X_umap'][mask, 1],
                  label=diag, alpha=0.6, s=50)
    ax.set_title('A) By Diagnosis')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Braak
    ax = axes[0, 1]
    if 'braaksc' in metadata_df.columns:
        scatter = ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                            c=metadata_df['braaksc'].values, cmap='viridis', s=50, alpha=0.6)
        ax.set_title('B) By Braak Stage')
        plt.colorbar(scatter, ax=ax)
    ax.grid(alpha=0.3)

    # Panel 3: MMSE
    ax = axes[1, 0]
    if 'mmse' in metadata_df.columns:
        scatter = ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                            c=metadata_df['mmse'].values, cmap='coolwarm', s=50, alpha=0.6)
        ax.set_title('C) By MMSE Score')
        plt.colorbar(scatter, ax=ax)
    ax.grid(alpha=0.3)

    # Panel 4: Pseudotime
    ax = axes[1, 1]
    scatter = ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                        c=adata.obs['dpt_pseudotime'].values, cmap='plasma', s=50, alpha=0.6)
    ax.set_title('D) By Pseudotime')
    plt.colorbar(scatter, ax=ax)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_file = f"{results_dir}/pseudotime_umap.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: pseudotime_umap.png")


def main(data_dir="data", results_dir="results", test_mode=False):
    """
    Run Step 1C: Disease pseudotime computation.

    Returns:
        dict: Results with keys: pseudotime_min, pseudotime_max, status
    """
    logger = logging.getLogger("Step1C")

    try:
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        logger.info("="*70)
        logger.info("STEP 1C: Disease Pseudotime")
        logger.info("="*70)

        # Load data
        logger.info("[1/6] Loading data...")
        proteomics_df, metadata_df, proportions_df = load_merged_data(processed_dir)

        # Feature selection
        logger.info("[2/6] Feature selection...")
        n_top = 200 if test_mode else 500
        proteomics_subset = select_top_variable_proteins(proteomics_df, n_top=n_top)

        # PCA
        logger.info("[3/6] PCA...")
        pca_coords, pca = compute_pca(proteomics_subset, n_components=50)

        # UMAP
        logger.info("[4/6] UMAP...")
        umap_coords, adata = compute_umap(pca_coords, n_neighbors=15, min_dist=0.3)
        adata.obs_names = proteomics_df.index

        # DPT root selection
        logger.info("[5/6] Pseudotime computation...")
        root = select_dpt_root(adata, metadata_df)
        pseudotime = compute_pseudotime(adata, root, n_dcs=10)
        adata.obs['dpt_pseudotime'] = pseudotime

        # Validate
        logger.info("[6/6] Validation...")
        is_valid = validate_pseudotime(pseudotime, metadata_df, logger)

        # Save
        logger.info("Saving outputs...")
        save_pseudotime_data(pseudotime, adata, metadata_df, proportions_df, processed_dir)
        plot_pseudotime_umap(adata, metadata_df, results_1_dir)

        logger.info("="*70)
        logger.info("STEP 1C COMPLETE")
        logger.info("="*70)

        return {
            'pseudotime_min': float(pseudotime.min()),
            'pseudotime_max': float(pseudotime.max()),
            'pseudotime_mean': float(pseudotime.mean()),
            'valid': is_valid,
            'status': 'PASS' if is_valid else 'WARNING'
        }

    except Exception as e:
        logger.error(f"Step 1C failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
