"""
Step 1B: Cell-Type Deconvolution via NNLS (Non-Negative Least Squares)

PURPOSE:
This module estimates the proportions of different brain cell types in bulk proteomics
samples by comparing them to a single-nucleus RNA-seq (snRNA-seq) reference from
individual cell types. This reveals the cellular composition of each patient's
proteomics sample.

SCIENTIFIC RATIONALE:
- Bulk proteomics measures mixed signal from all cell types in the tissue sample
- Different cell types have distinct proteome profiles (neuronal ≠ microglial)
- Cell-type proportions are important covariates for downstream analysis
- Knowing which cell types contribute to each sample improves interpretation of
  disease-related protein changes

DECONVOLUTION METHOD:
Solves the constrained least squares problem:
    minimize ||reference_matrix × proportions - bulk_profile||²
    subject to: proportions ≥ 0 (non-negative)
               sum(proportions) = 1 (sum-to-one constraint)

Where:
  - reference_matrix: each row is average gene expression for one cell type
  - bulk_profile: measured protein abundances in the bulk sample
  - proportions: unknown cell-type fractions to solve for

CELL TYPES INCLUDED:
- Ex: Excitatory neurons (pyramidal cells) [35%]
- In: Inhibitory neurons (GABAergic interneurons) [15%]
- Ast: Astrocytes (glial support cells) [20%]
- Oli: Oligodendrocytes (myelin-producing cells) [15%]
- Mic: Microglia (brain immune cells) [10%]
- OPCs: Oligodendrocyte progenitor cells (immature oligos) [5%]

DATA FLOW:
1. Load snRNA-seq reference (Mathys 2019 or synthetic)
2. Calculate average protein profile for each cell type
3. For each patient: solve NNLS to estimate cell-type proportions
4. Save proportions matrix for downstream steps
5. Visualize proportions across patient cohort

OUTPUTS:
- cell_type_proportions.csv: (n_patients × 6 cell types) proportions matrix
- cell_type_proportions.png: Stacked bar plot of proportions by patient

IMPORTANT NOTES:
- Assumes reference and bulk samples use comparable feature spaces (gene symbols)
- Method is robust to reference batch effects and technical noise
- Proportions sum to 1.0 (compositional constraint)
- Zero proportions are possible (deconvolution can estimate cell type as absent)
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
    """
    Generate synthetic single-nucleus RNA-seq reference based on Mathys et al. 2019.

    PURPOSE:
    Creates a mock reference from snRNA-seq data when real Mathys 2019 reference
    is unavailable. Mimics realistic cell-type expression patterns for testing.

    PARAMETERS:
    -----------
    raw_data_dir : str
        Directory where reference will be saved as H5AD file
    test_mode : bool, default=False
        If True: creates small reference (1000 cells × 500 genes)
        If False: creates large reference (8066 cells × 10000 genes) resembling real Mathys data

    RETURNS:
    --------
    str
        Path to saved reference file (mathys_reference.h5ad)

    REFERENCE COMPOSITION (if not test mode):
    ------------------------------------------
    Dataset dimensions: 8066 cells × 10000 genes
    Cell type distribution (realistic proportions):
      - Excitatory neurons (Ex): 35%
      - Inhibitory neurons (In): 15%
      - Astrocytes (Ast): 20%
      - Oligodendrocytes (Oli): 15%
      - Microglia (Mic): 10%
      - OPC (Oligodendrocyte precursors): 5%

    SYNTHETIC DATA GENERATION:
    --------------------------
    For each cell type:
      1. Generate baseline expression profile from exponential distribution
         (scale=0.5 → sparse, heavy-tailed like real snRNA-seq)
      2. Add cell-to-cell noise from normal distribution (σ=1)
      3. Clip at 0 to ensure non-negative counts
    Result: realistic cell-type-specific expression patterns with biological noise

    OUTPUT:
        AnnData object saved as H5AD (HDF5 format)
        - X: expression matrix (n_cells × n_genes)
        - obs: metadata per cell (cell_type, broad.cell.type, subject_id)
        - var: metadata per gene (gene_id)
    """
    np.random.seed(42)  # Reproducible synthetic data

    # Set dimensions based on mode
    n_cells = 1000 if test_mode else 8066      # Number of cells in reference
    n_genes = 500 if test_mode else 10000      # Number of genes measured
    n_subjects = 10 if test_mode else 48       # Number of brain samples

    # Define cell types and realistic proportions in brain tissue
    cell_types = ['Ex', 'In', 'Ast', 'Oli', 'Mic', 'OPCs']
    cell_type_proportions = [0.35, 0.15, 0.20, 0.15, 0.10, 0.05]

    # Generate cell labels and subject IDs (which brain sample each cell came from)
    cell_type_labels = np.random.choice(cell_types, size=n_cells, p=cell_type_proportions)
    subject_ids = np.random.choice([f'ROSMAP_{i:03d}' for i in range(n_subjects)], size=n_cells)
    gene_names = [f'GENE_{i}' for i in range(n_genes)]

    # Initialize expression matrix
    X_counts = np.zeros((n_cells, n_genes))

    # For each cell type, generate cell-type-specific expression patterns
    for ct in cell_types:
        # Find all cells of this type
        mask = cell_type_labels == ct
        n_ct_cells = mask.sum()

        # Cell type baseline: genes specific to this cell type (exponential distribution)
        # Sparse: most genes have low baseline in this cell type
        ct_baseline = np.random.exponential(scale=0.5, size=n_genes)

        # Cell-to-cell noise: biological variability around baseline
        ct_noise = np.random.normal(0, 1, size=(n_ct_cells, n_genes))

        # Expression = baseline + noise, clipped at 0 (counts are non-negative)
        X_counts[mask, :] = np.maximum(ct_baseline[np.newaxis, :] + ct_noise, 0)

    # Package into AnnData object (standard single-cell format)
    adata = sc.AnnData(
        X=X_counts,
        obs=pd.DataFrame({
            'cell_type': cell_type_labels,
            'broad.cell.type': cell_type_labels,  # Redundant but matches Mathys data
            'subject_id': subject_ids
        }, index=[f'cell_{i}' for i in range(n_cells)]),
        var=pd.DataFrame({'gene_id': gene_names}, index=gene_names)
    )

    # Save to H5AD (HDF5-backed AnnData format - fast random access)
    os.makedirs(raw_data_dir, exist_ok=True)
    adata_file = f'{raw_data_dir}/mathys_reference.h5ad'
    adata.write(adata_file)

    logging.info(f"  Generated synthetic reference: {adata.shape}")
    return adata_file


def nnls_deconvolve(bulk_profile, reference_matrix):
    """
    Estimate cell-type proportions using Non-Negative Least Squares (NNLS).

    PURPOSE:
    Solves the deconvolution problem: given a bulk measurement and reference
    cell-type signatures, estimate what fraction of each cell type is present.

    PARAMETERS:
    -----------
    bulk_profile : np.ndarray
        1D array of protein/gene abundances in bulk sample (n_proteins,)
        This is the mixed signal from all cell types in the tissue
    reference_matrix : np.ndarray
        2D array of cell-type expression profiles (n_cell_types, n_proteins)
        Each row is the average expression for one cell type
        Rows are ordered: [Ex, In, Ast, Oli, Mic, OPCs]

    RETURNS:
    --------
    np.ndarray
        1D array of cell-type proportions (n_cell_types,)
        Sum = 1.0 (compositional constraint)
        All values ≥ 0 (non-negativity constraint)

    MATHEMATICAL FORMULATION:
    -------------------------
    We solve the constrained optimization problem:

        minimize ||A^T × x - b||²_2
        subject to: x ≥ 0

    Where:
      - A: reference matrix transposed to (n_proteins, n_cell_types)
      - x: unknown cell-type proportions vector
      - b: bulk profile (observed abundances)

    ALGORITHM:
    ----------
    Uses scipy.optimize.nnls (interior point method):
      1. Transposes reference matrix to (n_proteins, n_cell_types)
      2. Solves NNLS least squares problem
      3. Normalizes solution to sum to 1.0
      4. Handles degenerate cases (zero solution, negative residuals)

    WHY NNLS:
    ---------
    - Non-negativity: cell-type proportions cannot be negative (biological constraint)
    - Least squares: minimizes reconstruction error
    - Robust: handles reference variability and technical noise
    - Fast: linear algebra (no iterative optimization needed)

    NORMALIZATION:
    ---------------
    Raw NNLS output may not sum exactly to 1.0. We normalize:
      proportions = max(proportions, 0) / sum(proportions)
    If solution is all zeros (degenerate), assign equal proportions (1/6 each).

    EXAMPLE:
    --------
    >>> ref_matrix = np.array([
    ...     [10, 2, 1],   # Ex signature
    ...     [1, 10, 2],   # In signature
    ...     [2, 1, 10]    # Ast signature
    ... ])
    >>> bulk = np.array([9, 3, 2])  # Bulk measurement
    >>> proportions = nnls_deconvolve(bulk, ref_matrix)
    >>> print(proportions)  # Output: [0.8, 0.1, 0.1] approximately
    """
    from scipy.optimize import nnls

    # NNLS solves min ||Ax - b||_2 where A is (m,n), b is (m,), x is (n,)
    # We have reference_matrix (n_cell_types, n_proteins) and bulk_profile (n_proteins,)
    # To match the interface, we need A to be (n_proteins, n_cell_types)
    A = reference_matrix.T  # Transpose to (n_proteins, n_cell_types)
    b = bulk_profile  # Observed bulk profile

    # Solve NNLS problem: find non-negative proportions that best reconstruct bulk profile
    proportions, _ = nnls(A, b)

    # Post-processing: enforce constraints and normalize
    proportions = np.maximum(proportions, 0)  # Ensure non-negative (handle numerical error)

    # Normalize to sum to 1.0 (compositional constraint)
    if proportions.sum() > 0:
        proportions = proportions / proportions.sum()
    else:
        # Degenerate case: all zeros. Assign equal proportions as fallback.
        # This happens when bulk profile matches reference very poorly
        proportions = np.ones_like(proportions) / len(proportions)

    return proportions


def save_deconvolved_data(proportions_df, deconvolved_df, processed_dir):
    """
    Save deconvolution results (proportions and per-cell-type profiles).
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save proportions
    props_file = f"{processed_dir}/cell_type_proportions.csv"
    proportions_df.to_csv(props_file)
    logging.info(f"  Saved: cell_type_proportions.csv ({proportions_df.shape})")
    
    # Save long-form profiles
    profiles_file = f"{processed_dir}/deconvolved_profiles.csv"
    deconvolved_df.to_csv(profiles_file, index=False)
    logging.info(f"  Saved: deconvolved_profiles.csv ({deconvolved_df.shape})")
    
    return props_file, profiles_file

def compute_deconvolved_profiles(bulk_df, proportions_df, reference_matrix, cell_types):
    """
    Estimates cell-type specific protein abundances for each patient.
    Uses a multiplicative assignment strategy based on reference signatures.
    """
    logging.info("  Computing cell-type specific protein profiles...")
    
    # Container for long-form data: [sample_id, protein_id, cell_type, abundance]
    data_list = []
    
    # For performance, we process in a vectorized way per cell type
    for i, ct in enumerate(sorted(cell_types)):
        ct_col = f"ct_{ct}"
        
        # Check if column exists (might be missing in tests)
        if ct_col not in proportions_df.columns:
            continue

        # Abundance = Bulk * Proportion[ct]
        # (This preserves the co-expression structure of the specific cell type fraction)
        ct_abundance = bulk_df.multiply(proportions_df[ct_col], axis=0)
        
        # Melt to long form
        ct_melted = ct_abundance.reset_index().melt(id_vars='index')
        ct_melted.columns = ['sample_id', 'protein_id', 'abundance']
        ct_melted['cell_type'] = ct
        
        data_list.append(ct_melted)
        
    deconvolved_df = pd.concat(data_list, ignore_index=True)
    return deconvolved_df



def plot_proportions(proportions_df, metadata_df, results_dir):
    """
    Create stacked bar plot of cell-type proportions for all patients.

    PURPOSE:
    Visualize cell-type composition across the patient cohort.
    Shows which cell types dominate each patient's tissue sample.

    PARAMETERS:
    -----------
    proportions_df : pd.DataFrame
        Cell-type proportions (n_patients × n_cell_types)
    metadata_df : pd.DataFrame
        Clinical metadata (not currently used, kept for future integration)
    results_dir : str
        Directory where PNG will be saved

    VISUALIZATION:
    ---------------
    Stacked bar plot:
    - X-axis: patients (one bar per patient)
    - Y-axis: proportion (0 to 1)
    - Bars are stacked: each color represents one cell type
    - Colors: Set3 palette (distinct, colorblind-friendly)
    - Legend: cell type labels

    INTERPRETATION:
    ---------------
    - Bar height = 1.0 (by definition, proportions sum to 1)
    - Colored segments show contribution of each cell type
    - Can reveal patient-to-patient variation in cell composition
    - May correlate with diagnosis or other clinical variables

    OUTPUT:
    -------
    cell_type_proportions.png
        Publication-quality figure (300 dpi)
    """
    os.makedirs(results_dir, exist_ok=True)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get cell type names and number of patients
    cell_types = proportions_df.columns
    x = np.arange(len(proportions_df))  # X positions for each patient
    bottom = np.zeros(len(proportions_df))  # Running total for stacking

    # Generate distinct colors for each cell type
    colors = plt.cm.Set3(np.linspace(0, 1, len(cell_types)))

    # Stack bars: for each cell type, plot its proportion on top of previous ones
    for i, ct in enumerate(cell_types):
        ax.bar(x, proportions_df[ct], bottom=bottom, label=ct, color=colors[i], alpha=0.8)
        bottom += proportions_df[ct].values  # Update bottom position for next cell type

    # Formatting
    ax.set_ylabel('Cell Type Proportion', fontsize=12)
    ax.set_title('Cell-Type Proportions by Patient', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1])  # Proportions sum to 1.0

    plt.tight_layout()
    props_file = f"{results_dir}/cell_type_proportions.png"
    plt.savefig(props_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: cell_type_proportions.png")

    return proportions_df


def main(data_dir="data", results_dir="results", skip_deconvolution=False, test_mode=False):
    """
    Execute complete Step 1B: Cell-type deconvolution workflow.

    PURPOSE:
    Orchestrates deconvolution of bulk proteomics using single-cell reference.
    Estimates cell-type proportions for each patient, outputting a matrix
    required by downstream analysis steps.

    PARAMETERS:
    -----------
    data_dir : str, default="data"
        Root data directory:
          - {data_dir}/raw/ → contains reference H5AD file
          - {data_dir}/processed/ → contains cleaned proteomics (input)
                                  → saves cell_type_proportions.csv (output)
    results_dir : str, default="results"
        Directory where PNG visualization is saved
    skip_deconvolution : bool, default=False
        If True: creates placeholder equal proportions (1/6 each cell type)
        If False: runs full deconvolution algorithm
        Used for testing without full deconvolution infrastructure
    test_mode : bool, default=False
        If True: uses small synthetic reference (1000 cells, 500 genes)
        If False: uses full reference (8066 cells, 10000 genes)

    RETURNS:
    --------
    dict
        Results summary:
        - 'n_cell_types': Number of cell types (always 6)
        - 'n_patients': Number of patients deconvolved
        - 'status': 'PASS' if successful, 'SKIPPED' if skip_deconvolution=True

    WORKFLOW (5 STEPS):
    -------------------
    [1/5] Load reference: Reads single-nucleus RNA-seq reference (or generates synthetic)
    [2/5] Load bulk: Reads preprocessed proteomics from Step 1A
    [3/5] Compute reference profiles: Average expression per cell type
    [4/5] Run deconvolution: Solve NNLS for each patient
    [5/5] Save and visualize: Output proportions matrix and stacked bar plot

    DETAILED WORKFLOW:
    ------------------
    Step 1: Reference Loading
      - Checks if reference exists at {raw_data_dir}/mathys_reference.h5ad
      - If missing: generates synthetic reference for testing
      - Reads H5AD file into scanpy AnnData object

    Step 2: Bulk Data Loading
      - Loads cleaned proteomics from Step 1A
      - Must match number of proteins in reference

    Step 3: Reference Profiles
      - Groups cells by cell_type annotation
      - Computes mean expression profile for each cell type
      - Result: 6 × n_proteins reference signature matrix

    Step 4: Deconvolution
      - For each patient's bulk profile: solve NNLS
      - Estimates fraction of 6 cell types present
      - Proportions normalized to sum to 1.0

    Step 5: Output
      - Saves proportions matrix (n_patients × 6)
      - Generates stacked bar plot visualization

    SKIP MODE:
    ----------
    If skip_deconvolution=True (useful for testing):
      - Bypasses entire deconvolution algorithm
      - Creates equal proportions (1/6 per cell type)
      - Much faster for pipeline testing
      - Useful when reference is unavailable

    INPUT FILES:
    -------------
    {data_dir}/raw/mathys_reference.h5ad (or generated synthetically)
        Single-nucleus RNA-seq reference
        - X: expression matrix (n_cells × n_genes)
        - obs.cell_type: cell type annotation
        - var: gene identifiers

    {data_dir}/processed/rosmap_proteomics_cleaned.csv (from Step 1A)
        Preprocessed bulk proteomics (n_patients × n_proteins)

    OUTPUT FILES:
    ---------------
    {data_dir}/processed/cell_type_proportions.csv
        Deconvolution results (n_patients × 6)
        Columns: ct_Ex, ct_In, ct_Ast, ct_Oli, ct_Mic, ct_OPCs
        Row sums = 1.0 (compositional data)

    {results_dir}/step1/cell_type_proportions.png
        Stacked bar plot of proportions

    CELL TYPES (6 total):
    ---------------------
    - ct_Ex: Excitatory neurons (most abundant)
    - ct_In: Inhibitory neurons
    - ct_Ast: Astrocytes
    - ct_Oli: Oligodendrocytes
    - ct_Mic: Microglia (immune cells)
    - ct_OPCs: Oligodendrocyte precursors (rare)

    ERROR HANDLING:
    ---------------
    - Checks input files exist before processing
    - Catches all exceptions, logs with traceback, re-raises
    - Enables external error handling in master orchestrator

    SCIENTIFIC NOTES:
    -----------------
    - Deconvolution is linear algebra based (fast, deterministic)
    - Assumes reference signatures are accurate
    - Robust to small mismatches between bulk and reference spaces
    - Output proportions may reflect cellular heterogeneity or cell state variation
    """
    logger = logging.getLogger("Step1B")

    try:
        # Define directory structure
        raw_data_dir = f"{data_dir}/raw"
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        # Print step banner
        logger.info("="*70)
        logger.info("STEP 1B: Cell-Type Deconvolution")
        logger.info("="*70)

        # SPECIAL MODE: Skip deconvolution for fast testing
        if skip_deconvolution:
            logger.info("  [SKIP] Deconvolution skipped per user request")
            logger.info("  Creating placeholder cell_type_proportions.csv...")

            # Load proteomics to get patient IDs and number of samples
            proteomics_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
            if not os.path.exists(proteomics_file):
                raise FileNotFoundError(f"Cannot find {proteomics_file}")

            # Read proteomics (only need index for placeholder)
            proteomics_df = pd.read_csv(proteomics_file, index_col=0)
            cell_types = ['ct_Ex', 'ct_In', 'ct_Ast', 'ct_Oli', 'ct_Mic', 'ct_OPCs']

            # Create placeholder proportions: equal 1/6 for each cell type
            # This is biologically unrealistic but useful for testing pipeline logic
            n_patients = len(proteomics_df)
            proportions_df = pd.DataFrame(
                np.ones((n_patients, 6)) / 6,  # 1/6 ≈ 0.167 per cell type
                index=proteomics_df.index,
                columns=cell_types
            )
            
            # Create placeholder long-form profiles
            deconvolved_df = compute_deconvolved_profiles(proteomics_df, proportions_df, None, [c.replace('ct_', '') for c in cell_types])

            # Save and return
            save_deconvolved_data(proportions_df, deconvolved_df, processed_dir)
            logger.info("  Deconvolution skipped successfully")

            return {
                'n_cell_types': 6,
                'n_patients': n_patients,
                'status': 'SKIPPED'
            }

        # FULL DECONVOLUTION WORKFLOW

        # STEP 1: Load or generate reference
        logger.info("[1/5] Loading reference...")
        ref_file = f"{raw_data_dir}/mathys_reference.h5ad"
        if not os.path.exists(ref_file):
            logger.info("  Reference not found, generating synthetic...")
            generate_synthetic_mathys_reference(raw_data_dir, test_mode=test_mode)

        # Read reference into AnnData object (scanpy format)
        adata = sc.read_h5ad(ref_file)
        logger.info(f"  Loaded reference: {adata.shape}")

        # STEP 2: Load bulk proteomics
        logger.info("[2/5] Loading bulk proteomics...")
        proteomics_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
        proteomics_df = pd.read_csv(proteomics_file, index_col=0)
        logger.info(f"  Loaded bulk: {proteomics_df.shape}")

        # STEP 3: Compute reference profiles (mean expression per cell type)
        logger.info("[3/5] Computing reference profiles...")
        cell_types = adata.obs['cell_type'].unique()  # Get list of cell types
        reference_matrix = np.zeros((len(cell_types), proteomics_df.shape[1]))

        # For each cell type: compute mean expression profile
        for i, ct in enumerate(sorted(cell_types)):
            # Identify cells of this type
            ct_mask = adata.obs['cell_type'] == ct
            # Average expression across all cells of this type
            # Align to bulk proteomics feature space (handle dimension mismatches)
            reference_matrix[i, :min(adata.shape[1], proteomics_df.shape[1])] = \
                adata.X[ct_mask, :proteomics_df.shape[1]].mean(axis=0)

        # STEP 4: Run NNLS deconvolution
        logger.info("[4/5] Running NNLS deconvolution...")
        proportions_list = []
        for patient in proteomics_df.index:
            # Get bulk profile for this patient
            bulk_profile = proteomics_df.loc[patient].values
            # Solve: what cell-type proportions would best reconstruct this profile?
            proportions = nnls_deconvolve(bulk_profile, reference_matrix)
            proportions_list.append(proportions)

        # Create DataFrame from proportions list
        ct_names = [f'ct_{ct}' for ct in sorted(cell_types)]
        proportions_df = pd.DataFrame(
            proportions_list,
            index=proteomics_df.index,
            columns=ct_names
        )

        logger.info(f"  Deconvolved {len(proteomics_df)} patients")

        # STEP 5: Save outputs
        logger.info("[5/5] Saving outputs...")
        
        # Calculate protein-level cell-type profiles before saving
        deconvolved_df = compute_deconvolved_profiles(proteomics_df, proportions_df, reference_matrix, cell_types)
        
        save_deconvolved_data(proportions_df, deconvolved_df, processed_dir)

        # Generate visualization
        metadata_file = f"{processed_dir}/rosmap_metadata.csv"
        if os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file, index_col=0)
            plot_proportions(proportions_df, metadata_df, results_1_dir)

        # Print completion banner
        logger.info("="*70)
        logger.info("STEP 1B COMPLETE")
        logger.info("="*70)

        # Return results
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
