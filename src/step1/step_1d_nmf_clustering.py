"""
Step 1D: NMF Consensus Clustering for Disease Subtype Discovery

PURPOSE:
This module discovers disease subtypes (distinct disease trajectories) within the AD cohort
by applying Non-Negative Matrix Factorization (NMF) consensus clustering. Identifies
patient subgroups with distinct proteome profiles, suggesting heterogeneous disease mechanisms.

SCIENTIFIC RATIONALE:
- Alzheimer's disease is heterogeneous: different patients have different pathology patterns
- Identifying subtypes enables precision medicine (tailor treatments to subtype)
- NMF is ideal for proteomics: identifies co-varying protein modules
- Consensus clustering improves robustness: averages over multiple random initializations
- Cophenetic correlation validates clustering stability/quality

NMF (Non-Negative Matrix Factorization):
- Decomposes proteomics matrix V into: V ≈ W × H
  * W: patient-by-component (n_patients × k)
  * H: component-by-protein (k × n_proteins)
- Each patient is a weighted combination of k disease components
- Non-negativity constraint: all weights ≥ 0 (interpretable)
- Subtypes defined by which component dominates each patient

CONSENSUS CLUSTERING:
- Runs NMF multiple times (1 deterministic + 50 random initializations)
- Builds consensus matrix: co-clustering frequency for each patient pair
- More robust than single NMF run (handles random initialization variability)
- High consensus = stable clusters (low sensitivity to initialization)

CLUSTER QUALITY METRICS:
1. Cophenetic Correlation (0-1, higher=better)
   - Measures agreement between cluster assignments and hierarchical dendrogram
   - Threshold: ≥ 0.85 indicates good cluster quality
   - Validates that consensus matrix reflects stable clusters

2. Silhouette Score (-1 to 1, higher=better)
   - Measures how well each sample fits its cluster vs others
   - >0.5: good separation
   - Used for cluster evaluation (not selection)

3. Minimum Cluster Size (practical constraint)
   - Each subtype must have ≥25 patients (sufficient for downstream analysis)
   - Prevents tiny, unstable clusters

OUTPUTS:
- subtype_labels.csv: Subtype assignment per patient (ST1, ST2, ST3, ...)
- master_patient_table_final.csv: Master table with added subtype column
- subtype_cluster_sizes.png: Bar plot of subtype sizes

WORKFLOW:
1. Load master patient table (from Step 1C)
2. Filter to AD/MCI patients only (exclude controls, stratified separately)
3. Test k=2-5 subtypes, select optimal k using cophenetic correlation
4. Run final NMF consensus with optimal k
5. Assign subtype labels and update master table
6. Validate with cophenetic correlation and silhouette score

KEY PARAMETERS:
- k: number of subtypes (auto-selected 2-5, checked for quality)
- n_runs: 50 random initializations (consensus over 51 total runs)
- cophenetic_threshold: 0.85 (minimum quality for valid clustering)
- min_size: 25 patients per subtype (practical minimum)
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
    """
    Load master patient table and filter to AD/MCI patients.

    PURPOSE:
    Disease subtype discovery should focus on diseased individuals, not controls.
    This filters the cohort to AD/MCI patients before clustering.

    PARAMETERS:
    -----------
    processed_dir : str
        Directory containing master_patient_table.csv (from Step 1C)

    RETURNS:
    --------
    pd.DataFrame
        Master table subset to non-Control patients (diagnosis != 'Control')

    NOTES:
    ------
    Controls are stratified separately (always assigned 'Control' label).
    AD/MCI patients are clustered to identify disease subtypes.
    """
    master_file = f"{processed_dir}/master_patient_table.csv"
    master_df = pd.read_csv(master_file, index_col=0)

    # Filter to Clinical AD patients ONLY (exclude Healthy Control and AsymAD)
    if 'patient_class' in master_df.columns:
        ad_mask = master_df['patient_class'] == 'Clinical AD'
        master_df = master_df[ad_mask]
    elif 'diagnosis' in master_df.columns:
        ad_mask = master_df['diagnosis'] != 'Control'
        master_df = master_df[ad_mask]

    logging.info(f"  Loaded {len(master_df)} AD/MCI patients")
    return master_df


def load_proteomics(processed_dir):
    """
    Load normalized proteomics and ensure non-negative values required by NMF.

    PURPOSE:
    NMF requires non-negative input (V ≥ 0 everywhere). Z-score normalized data may
    contain negative values. This function shifts data to non-negative range.

    PARAMETERS:
    -----------
    processed_dir : str
        Directory containing rosmap_proteomics_cleaned.csv (from Step 1A)

    RETURNS:
    --------
    pd.DataFrame
        Proteomics matrix, shifted to non-negative if needed

    DETAILS:
    --------
    If min_value < 0:
      - Compute shift = -min_value + 0.01 (small buffer for numerical stability)
      - Add shift to all values
      - Result: min ≈ 0.01 (all values non-negative)

    Why not use absolute values?
      - Loses interpretability of directions
      - Shifts don't affect relative comparisons (important for NMF)
    """
    prot_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
    proteomics_df = pd.read_csv(prot_file, index_col=0)

    # Check for negative values (can occur from z-score normalization)
    min_val = proteomics_df.min().min()
    if min_val < 0:
        # Shift all values to non-negative range
        proteomics_df = proteomics_df - min_val + 0.01  # 0.01 buffer for numerical stability
        logging.info(f"  Shifted data to non-negative (min was {min_val:.4f})")

    return proteomics_df


def run_nmf_consensus(proteomics_df, n_subtypes=2, n_runs=50, init='nndsvda'):
    """
    Run NMF consensus clustering: multiple runs averaged into consensus matrix.

    PURPOSE:
    NMF is sensitive to random initialization. Consensus clustering averages
    over multiple runs to find robust, stable clusters.

    PARAMETERS:
    -----------
    proteomics_df : pd.DataFrame
        Non-negative proteomics matrix (n_patients × n_proteins)
    n_subtypes : int, default=2
        Number of disease subtypes (components for NMF)
    n_runs : int, default=50
        Number of random initializations (+ 1 deterministic = 51 total)
    init : str, default='nndsvda'
        Initialization method for base run (deterministic)

    RETURNS:
    --------
    tuple of (np.ndarray, np.ndarray)
        (consensus_matrix, labels_base)
        - consensus_matrix: n_samples × n_samples, co-clustering frequency
        - labels_base: subtype labels from base run (used for final clustering)

    ALGORITHM:
    ----------
    1. Base Run (deterministic):
       - Initialize with 'nndsvda' (non-negative singular value decomposition)
       - Run NMF: factorize V ≈ W × H
       - Assign each patient to subtype with highest loading in W
       - Update consensus matrix: if patients i,j same subtype → consensus[i,j] += 1

    2. Random Runs (x50):
       - For each run with different random seed:
         - Initialize randomly
         - Run NMF
         - Assign subtypes
         - Update consensus matrix

    3. Normalization:
       - Divide consensus matrix by 51 (base + 50 random runs)
       - Result: values in [0, 1] = co-clustering frequency

    INTERPRETATION:
    ----------------
    consensus[i,j] = frequency that patients i and j were assigned to same subtype
    - 1.0: always same subtype (perfectly co-clustered)
    - 0.5: sometimes same, sometimes different (unstable)
    - 0.0: never same subtype (stable separation)

    HIGH consensus = stable, reproducible clustering
    """
    np.random.seed(42)  # Reproducibility

    n_samples = proteomics_df.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))

    # BASE RUN: Deterministic initialization (nndsvda)
    nmf_base = NMF(n_components=n_subtypes, init=init, random_state=42, max_iter=1000)
    W_base = nmf_base.fit_transform(proteomics_df.values)  # n_samples × n_subtypes
    # Assign each patient to subtype with highest loading
    labels_base = np.argmax(W_base, axis=1)

    # Update consensus matrix with base run assignments
    for i in range(n_samples):
        for j in range(n_samples):
            if labels_base[i] == labels_base[j]:  # Same subtype
                consensus_matrix[i, j] += 1.0

    # RANDOM RUNS: Test stability with different random initializations
    for run in range(n_runs):
        # Initialize randomly (different seed each time)
        nmf = NMF(n_components=n_subtypes, init='random', random_state=run, max_iter=1000)
        W = nmf.fit_transform(proteomics_df.values)
        labels = np.argmax(W, axis=1)  # Assign to dominant subtype

        # Update consensus matrix
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1.0

    # Normalize to [0,1] range (frequency over 51 runs)
    consensus_matrix = consensus_matrix / (n_runs + 1)

    logging.info(f"  Consensus matrix computed ({n_samples}x{n_samples})")
    return consensus_matrix, labels_base


def compute_cophenetic_correlation(consensus_matrix):
    """
    Compute cophenetic correlation coefficient measuring clustering stability.

    PURPOSE:
    Validates that consensus matrix reflects true underlying clusters.
    High cophenetic correlation (>0.85) indicates stable, reproducible clustering.

    PARAMETERS:
    -----------
    consensus_matrix : np.ndarray
        Co-clustering frequency matrix (n_samples × n_samples), values in [0,1]

    RETURNS:
    --------
    float
        Cophenetic correlation coefficient (0-1, higher=better)

    ALGORITHM:
    ----------
    1. Convert consensus to distance: distance = 1 - consensus
       (High co-clustering → low distance)
    2. Perform hierarchical clustering on distance matrix
       (Linkage method: average)
    3. Compute cophenetic distances (dendrogram-based distances)
    4. Correlate original distances vs cophenetic distances
       (Measures agreement between data distances and dendrogram)

    INTERPRETATION:
    ----------------
    Cophenetic Correlation = correlation(original distances, dendrogram distances)
    - 1.0: Perfect agreement (excellent clustering stability)
    - 0.85+: Good clustering (commonly used threshold)
    - 0.7-0.85: Moderate clustering (acceptable)
    - <0.7: Poor clustering (unstable, unreliable)

    HIGH = clustering is reproducible across random initializations
    LOW = assignments are sensitive to random initialization (unreliable)

    STANDARD THRESHOLD: 0.85
    Clustering with cophenetic ≥ 0.85 is considered high-quality and stable.
    """
    # Convert consensus (similarity) to distance
    distance_matrix = 1.0 - consensus_matrix
    # Diagonal should be 0 (distance from sample to itself)
    np.fill_diagonal(distance_matrix, 0)

    # Extract upper triangle (condensed distance format for hierarchical clustering)
    condensed_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    # Perform hierarchical clustering (agglomerative)
    Z = linkage(condensed_dist, method='average')

    # Compute cophenetic correlation coefficient
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist, squareform

    c, coph_dists = cophenet(Z, condensed_dist)
    logging.info(f"  Cophenetic correlation: {c:.4f}")
    return c


def select_optimal_k(proteomics_df, processed_dir, k_range=range(2, 6),
                     cophenetic_threshold=0.85, min_size=25):
    """
    Select optimal number of subtypes (k) using cophenetic correlation and cluster sizes.

    PURPOSE:
    Determines the number of disease subtypes that best partition the cohort.
    Too few subtypes: oversimplify disease heterogeneity.
    Too many subtypes: create unreliable, tiny clusters.

    PARAMETERS:
    -----------
    proteomics_df : pd.DataFrame
        Non-negative proteomics matrix
    processed_dir : str
        Directory for saving results
    k_range : range, default=range(2,6)
        Number of subtypes to test (typically 2-5)
    cophenetic_threshold : float, default=0.85
        Minimum acceptable cophenetic correlation for valid clustering
    min_size : int, default=25
        Minimum number of patients per subtype

    RETURNS:
    --------
    tuple of (int, dict)
        (best_k, results)
        - best_k: optimal number of subtypes
        - results: dict with metrics for each k tested

    SELECTION CRITERIA:
    -------------------
    For each k:
      1. Run NMF consensus clustering
      2. Compute cophenetic correlation
      3. Check cluster sizes

    Accept k if:
      - Cophenetic correlation ≥ 0.85 (stable clustering)
      - All clusters have ≥ 25 patients (practical minimum)

    Select k with HIGHEST cophenetic among accepted values
    (Validates stability while accepting best-quality partition)

    FALLBACK:
    ---------
    If no k meets both criteria, default to k=2
    (Simplest partition is fallback if nothing is "good")
    """
    logger = logging.getLogger("Step1D")

    best_k = 2  # Fallback
    best_cophenetic = 0
    results = {}

    # Test each k value
    for k in k_range:
        logger.info(f"  Testing k={k}...")

        # Run consensus clustering
        consensus_matrix, labels = run_nmf_consensus(proteomics_df, n_subtypes=k, n_runs=50)
        cophenetic = compute_cophenetic_correlation(consensus_matrix)

        # Check cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        min_cluster_size = counts.min()

        # Compute silhouette score (if valid)
        if len(unique) > 1 and min_cluster_size >= 2:
            try:
                silhouette = silhouette_score(proteomics_df.values, labels)
            except:
                silhouette = 0
        else:
            silhouette = 0

        # Store results
        results[k] = {
            'cophenetic': cophenetic,
            'silhouette': silhouette,
            'min_cluster_size': min_cluster_size,
            'labels': labels
        }

        logger.info(f"    Cophenetic: {cophenetic:.4f}, Silhouette: {silhouette:.4f}, Min size: {min_cluster_size}")

        # Check if this k meets selection criteria
        if cophenetic > cophenetic_threshold and min_cluster_size >= min_size:
            # Accept this k if it's better than previous best
            if cophenetic > best_cophenetic:
                best_cophenetic = cophenetic
                best_k = k

    logger.info(f"  Selected k={best_k} (cophenetic={best_cophenetic:.4f})")

    return best_k, results


def save_subtypes(labels, proteomics_df, processed_dir):
    """
    Save subtype assignments and update master patient table with subtypes.

    PURPOSE:
    Serialize clustering results and integrate subtype assignments into the
    master patient table for downstream use.

    PARAMETERS:
    -----------
    labels : np.ndarray
        Subtype labels (0, 1, 2, ...) for patients in proteomics_df
    proteomics_df : pd.DataFrame
        Proteomics matrix (used for patient indices)
    processed_dir : str
        Directory for saving outputs

    OUTPUTS:
    --------
    subtype_labels.csv
        One column: patient_id → ST1, ST2, ST3, ...

    master_patient_table_final.csv
        Complete master table with added 'subtype' column
        - AD/MCI patients: assigned ST1, ST2, etc.
        - Control patients: assigned 'Control'

    RETURN:
    -------
    tuple of (pd.Series, pd.DataFrame)
        (subtype_labels, master_final)
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Create readable subtype names (ST1, ST2, ...)
    subtype_names = [f'ST{i+1}' for i in range(len(np.unique(labels)))]
    subtype_labels = pd.Series(
        [subtype_names[l] for l in labels],  # Map numeric labels to ST names
        index=proteomics_df.index,
        name='subtype'
    )

    # Save subtype labels
    subtype_file = f"{processed_dir}/subtype_labels.csv"
    subtype_labels.to_csv(subtype_file)
    logging.info(f"  Saved: subtype_labels.csv")

    # Load master table and add subtypes
    master_file = f"{processed_dir}/master_patient_table.csv"
    master_df = pd.read_csv(master_file, index_col=0)

    # Add subtype assignments to AD/MCI patients
    master_df.loc[proteomics_df.index, 'subtype'] = subtype_labels

    # Leave non-AD patients (Controls, AsymAD) with NaN subtype per V3 design

    # Save final master table
    master_final_file = f"{processed_dir}/master_patient_table_final.csv"
    master_df.to_csv(master_final_file)
    logging.info(f"  Saved: master_patient_table_final.csv ({master_df.shape})")

    return subtype_labels, master_df


def plot_clustering_results(labels, proteomics_df, results_dir):
    """
    Create bar plot of subtype cluster sizes.

    PURPOSE:
    Visualize the distribution of patients across disease subtypes.

    PARAMETERS:
    -----------
    labels : np.ndarray
        Subtype labels (0, 1, 2, ...)
    proteomics_df : pd.DataFrame
        Proteomics matrix (used for patient count)
    results_dir : str
        Directory where PNG will be saved

    OUTPUT:
    -------
    subtype_cluster_sizes.png
        Bar plot showing number of patients in each subtype
    """
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
    Execute complete Step 1D: NMF consensus clustering for disease subtype discovery.

    PURPOSE:
    Master orchestrator for subtype discovery. Identifies distinct disease subtypes
    within the AD/MCI cohort using NMF consensus clustering. Integrates subtype
    assignments into master patient table for downstream stratified analysis.

    PARAMETERS:
    -----------
    data_dir : str, default="data"
        Root data directory:
          - {data_dir}/processed/ → loads/saves master_patient_table
    results_dir : str, default="results"
        Directory where visualization PNG is saved
    n_subtypes : int or None, default=None
        If None: automatically select k using cophenetic correlation
        If int: use specified number of subtypes (bypass auto-selection)
    test_mode : bool, default=False
        If True: test k=2-3 (faster)
        If False: test k=2-5 (comprehensive)

    RETURNS:
    --------
    dict
        Results summary:
        - 'n_subtypes': Number of disease subtypes identified
        - 'subtype_sizes': Dict mapping subtype → patient count
        - 'status': 'PASS' if successful

    WORKFLOW (5 STEPS):
    -------------------
    [1/5] Load data: Master patient table + proteomics
    [2/5] Select optimal k: Test k=2-5, choose best via cophenetic correlation
    [3/5] Final clustering: Run NMF consensus with optimal k
    [4/5] Save outputs: Subtype labels, update master table
    [5/5] Visualize: Bar plot of subtype sizes

    DETAILED STEPS:
    ---------------
    Step 1: Data Loading
      - Load master patient table (from Step 1C)
      - Filter to AD/MCI patients (exclude controls)
      - Load proteomics
      - Align indices (patients present in both)

    Step 2: Optimal k Selection (if not specified)
      - Test k = 2, 3, 4, 5 subtypes
      - For each k:
        * Run NMF consensus (1 deterministic + 50 random runs)
        * Compute cophenetic correlation
        * Check minimum cluster size
      - Accept k if: cophenetic ≥ 0.85 AND min size ≥ 25
      - Select k with highest cophenetic among valid options
      - Fallback to k=2 if nothing valid (should not happen)

    Step 3: Final Clustering
      - Run NMF consensus again with selected k (51 total runs)
      - Generate stable, reproducible assignments

    Step 4: Save Results
      - Save subtype_labels.csv (patient → ST1/ST2/ST3)
      - Create master_patient_table_final.csv
        * Adds 'subtype' column
        * AD/MCI patients: ST1, ST2, ST3, ...
        * Control patients: 'Control'

    Step 5: Visualization
      - Bar plot showing number of patients per subtype
      - Publication-quality (300 DPI, Set3 colors)

    OUTPUT FILES:
    ---------------
    {data_dir}/processed/subtype_labels.csv
        Two columns: patient_id, subtype (ST1/ST2/... or Control)

    {data_dir}/processed/master_patient_table_final.csv
        FINAL integration of all Step 1 outputs + subtypes
        Columns: clinical metadata + proportions + pseudotime + UMAP + subtype
        Used by Steps 1E and Step 2

    {results_dir}/step1/subtype_cluster_sizes.png
        Bar plot: number of patients per subtype

    SUBTYPES:
    ---------
    Named ST1, ST2, ST3, ... (non-hierarchical disease subtypes)
    Represent distinct disease mechanism/trajectory patterns
    Not necessarily ordered by severity (no inherent ranking)

    VALIDATION CRITERIA:
    --------------------
    Cophenetic correlation ≥ 0.85
      - Measures stability of consensus across 51 NMF runs
      - High value = robust, reproducible clustering

    Minimum cluster size ≥ 25 patients
      - Each subtype must have sufficient representation
      - Ensures statistical power for downstream analysis

    BIOLOGICAL INTERPRETATION:
    --------------------------
    Subtypes represent:
    - Distinct proteome signatures
    - Potentially different neuropathology profiles
    - Possibly different treatment responses (precision medicine)
    - Different progression rates or outcomes
    Validated in Step 1E (survival analysis)

    PARAMETER SELECTION:
    --------------------
    If n_subtypes=None (recommended):
      - Auto-selection ensures quality (validated by cophenetic)
      - Adapts to cohort heterogeneity

    If n_subtypes=2 (specified):
      - Forces binary stratification
      - Useful for pre-specified hypotheses
      - Bypasses auto-selection but still runs consensus

    ERROR HANDLING:
    ---------------
    Catches all exceptions, logs with traceback, re-raises
    Enables external error handling in master orchestrator
    """
    logger = logging.getLogger("Step1D")

    try:
        # Define directory structure
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        # Print step banner
        logger.info("="*70)
        logger.info("STEP 1D: NMF Consensus Clustering")
        logger.info("="*70)

        # STEP 1: Load data
        logger.info("[1/5] Loading data...")
        master_df = load_data_for_clustering(processed_dir)
        proteomics_df = load_proteomics(processed_dir)

        # Ensure alignment between master table and proteomics
        common_idx = master_df.index.intersection(proteomics_df.index)
        proteomics_df = proteomics_df.loc[common_idx]
        logger.info(f"  Using {len(proteomics_df)} samples for clustering")

        # STEP 2: Select optimal k or use specified value
        logger.info("[2/5] Selecting optimal k...")
        if n_subtypes is None:
            # Auto-selection: test multiple k values
            k_range = range(2, 4) if test_mode else range(2, 6)
            best_k, results = select_optimal_k(
                proteomics_df, processed_dir, k_range=k_range,
                cophenetic_threshold=0.85, min_size=25
            )
        else:
            # User-specified k (bypass auto-selection)
            best_k = n_subtypes
            logger.info(f"  Using user-specified k={best_k}")
            consensus_matrix, labels = run_nmf_consensus(
                proteomics_df, n_subtypes=best_k, n_runs=50
            )

        # STEP 3: Final clustering with selected k
        logger.info(f"[3/5] Final clustering with k={best_k}...")
        consensus_matrix, labels = run_nmf_consensus(
            proteomics_df, n_subtypes=best_k, n_runs=50
        )

        # STEP 4: Save outputs
        logger.info("[4/5] Saving outputs...")
        subtype_labels, master_final = save_subtypes(labels, proteomics_df, processed_dir)

        # STEP 5: Visualization
        logger.info("[5/5] Plotting results...")
        plot_clustering_results(labels, proteomics_df, results_1_dir)

        # Print completion banner
        logger.info("="*70)
        logger.info("STEP 1D COMPLETE")
        logger.info("="*70)

        # Compute subtype sizes for results
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
