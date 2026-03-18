"""
Step 1D: NMF Consensus Clustering for Subtype Discovery

Implements:
  - NMF consensus clustering (k=2-5)
  - Cophenetic correlation validation
  - Silhouette score evaluation
  - Optimal k selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
import os
import logging

warnings.filterwarnings('ignore')


def load_data_for_clustering(processed_dir):
    """Load master table for clustering."""
    master_file = f"{processed_dir}/master_patient_table.csv"
    master_df = pd.read_csv(master_file, index_col=0)

    # Use only AD/MCI patients (excluding Controls)
    if 'diagnosis' in master_df.columns:
        ad_mask = master_df['diagnosis'] != 'Control'
        master_df = master_df[ad_mask]

    logging.info(f"  Loaded {len(master_df)} AD/MCI patients")
    return master_df


def load_proteomics(processed_dir):
    """Load normalized proteomics."""
    prot_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
    proteomics_df = pd.read_csv(prot_file, index_col=0)

    # Ensure non-negative values (NMF requires this)
    # Shift all values if any negatives exist
    min_val = proteomics_df.min().min()
    if min_val < 0:
        proteomics_df = proteomics_df - min_val + 0.01  # Add small offset for numerical stability
        logging.info(f"  Shifted data to non-negative (min was {min_val:.4f})")

    return proteomics_df


def run_nmf_consensus(proteomics_df, n_subtypes=2, n_runs=50, init='nndsvda'):
    """Run NMF consensus clustering."""
    np.random.seed(42)

    n_samples = proteomics_df.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))

    # Base run
    nmf_base = NMF(n_components=n_subtypes, init=init, random_state=42, max_iter=1000)
    W_base = nmf_base.fit_transform(proteomics_df.values)
    labels_base = np.argmax(W_base, axis=1)

    # Update consensus
    for i in range(n_samples):
        for j in range(n_samples):
            if labels_base[i] == labels_base[j]:
                consensus_matrix[i, j] += 1.0

    # Random runs
    for run in range(n_runs):
        nmf = NMF(n_components=n_subtypes, init='random', random_state=run, max_iter=1000)
        W = nmf.fit_transform(proteomics_df.values)
        labels = np.argmax(W, axis=1)

        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1.0

    # Normalize
    consensus_matrix = consensus_matrix / (n_runs + 1)

    logging.info(f"  Consensus matrix computed ({n_samples}x{n_samples})")
    return consensus_matrix, labels_base


def compute_cophenetic_correlation(consensus_matrix):
    """Compute cophenetic correlation."""
    # Convert to distance
    distance_matrix = 1.0 - consensus_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Perform hierarchical clustering
    condensed_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    Z = linkage(condensed_dist, method='average')

    # Compute cophenetic correlation
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist, squareform

    c, coph_dists = cophenet(Z, condensed_dist)
    logging.info(f"  Cophenetic correlation: {c:.4f}")
    return c


def select_optimal_k(proteomics_df, processed_dir, k_range=range(2, 6),
                     cophenetic_threshold=0.85, min_size=25):
    """Select optimal k based on cophenetic correlation and silhouette score."""
    logger = logging.getLogger("Step1D")

    best_k = 2
    best_cophenetic = 0
    results = {}

    for k in k_range:
        logger.info(f"  Testing k={k}...")

        consensus_matrix, labels = run_nmf_consensus(proteomics_df, n_subtypes=k, n_runs=50)
        cophenetic = compute_cophenetic_correlation(consensus_matrix)

        # Check cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        min_cluster_size = counts.min()

        # Compute silhouette if valid
        if len(unique) > 1 and min_cluster_size >= 2:
            try:
                silhouette = silhouette_score(proteomics_df.values, labels)
            except:
                silhouette = 0
        else:
            silhouette = 0

        results[k] = {
            'cophenetic': cophenetic,
            'silhouette': silhouette,
            'min_cluster_size': min_cluster_size,
            'labels': labels
        }

        logger.info(f"    Cophenetic: {cophenetic:.4f}, Silhouette: {silhouette:.4f}, Min size: {min_cluster_size}")

        # Select this k if it meets criteria
        if cophenetic > cophenetic_threshold and min_cluster_size >= min_size:
            if cophenetic > best_cophenetic:
                best_cophenetic = cophenetic
                best_k = k

    logger.info(f"  Selected k={best_k} (cophenetic={best_cophenetic:.4f})")

    return best_k, results


def save_subtypes(labels, proteomics_df, processed_dir):
    """Save subtype labels and create master table with subtypes."""
    os.makedirs(processed_dir, exist_ok=True)

    # Create subtype labels
    subtype_names = [f'ST{i+1}' for i in range(len(np.unique(labels)))]
    subtype_labels = pd.Series(
        [subtype_names[l] for l in labels],
        index=proteomics_df.index,
        name='subtype'
    )

    # Save
    subtype_file = f"{processed_dir}/subtype_labels.csv"
    subtype_labels.to_csv(subtype_file)
    logging.info(f"  Saved: subtype_labels.csv")

    # Update master table
    master_file = f"{processed_dir}/master_patient_table.csv"
    master_df = pd.read_csv(master_file, index_col=0)

    master_df.loc[proteomics_df.index, 'subtype'] = subtype_labels
    # Add Control label for controls
    master_df.loc[master_df['subtype'].isna(), 'subtype'] = 'Control'

    master_final_file = f"{processed_dir}/master_patient_table_final.csv"
    master_df.to_csv(master_final_file)
    logging.info(f"  Saved: master_patient_table_final.csv ({master_df.shape})")

    return subtype_labels, master_df


def plot_clustering_results(labels, proteomics_df, results_dir):
    """Plot clustering results."""
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))

    for i, subtype in enumerate(sorted(unique)):
        mask = labels == subtype
        ax.bar(i, counts[subtype], color=colors[i], alpha=0.8, label=f'ST{subtype+1}')

    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_xlabel('Subtype', fontsize=12)
    ax.set_title('Subtype Cluster Sizes', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([f'ST{i+1}' for i in unique])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_file = f"{results_dir}/subtype_cluster_sizes.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: subtype_cluster_sizes.png")


def main(data_dir="data", results_dir="results", n_subtypes=None, test_mode=False):
    """
    Run Step 1D: NMF consensus clustering.

    Args:
        n_subtypes: Override auto-selection (default: None = auto-select)

    Returns:
        dict: Results with keys: n_subtypes, subtype_sizes, status
    """
    logger = logging.getLogger("Step1D")

    try:
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        logger.info("="*70)
        logger.info("STEP 1D: NMF Consensus Clustering")
        logger.info("="*70)

        # Load data
        logger.info("[1/5] Loading data...")
        master_df = load_data_for_clustering(processed_dir)
        proteomics_df = load_proteomics(processed_dir)

        # Match indices
        common_idx = master_df.index.intersection(proteomics_df.index)
        proteomics_df = proteomics_df.loc[common_idx]
        logger.info(f"  Using {len(proteomics_df)} samples for clustering")

        # Select k
        logger.info("[2/5] Selecting optimal k...")
        if n_subtypes is None:
            k_range = range(2, 4) if test_mode else range(2, 6)
            best_k, results = select_optimal_k(
                proteomics_df, processed_dir, k_range=k_range,
                cophenetic_threshold=0.85, min_size=25
            )
        else:
            best_k = n_subtypes
            logger.info(f"  Using user-specified k={best_k}")
            consensus_matrix, labels = run_nmf_consensus(
                proteomics_df, n_subtypes=best_k, n_runs=50
            )

        # Final clustering
        logger.info(f"[3/5] Final clustering with k={best_k}...")
        consensus_matrix, labels = run_nmf_consensus(
            proteomics_df, n_subtypes=best_k, n_runs=50
        )

        # Save
        logger.info("[4/5] Saving outputs...")
        subtype_labels, master_final = save_subtypes(labels, proteomics_df, processed_dir)

        # Plot
        logger.info("[5/5] Plotting results...")
        plot_clustering_results(labels, proteomics_df, results_1_dir)

        logger.info("="*70)
        logger.info("STEP 1D COMPLETE")
        logger.info("="*70)

        unique, counts = np.unique(labels, return_counts=True)
        subtype_sizes = {f'ST{i+1}': int(counts[i]) for i in range(len(unique))}

        return {
            'n_subtypes': best_k,
            'subtype_sizes': subtype_sizes,
            'status': 'PASS'
        }

    except Exception as e:
        logger.error(f"Step 1D failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
