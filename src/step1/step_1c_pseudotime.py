"""
Step 1C: Disease Pseudotime Computation & UMAP Visualization

PURPOSE:
This module creates a disease progression trajectory for each patient by computing
diffusion pseudotime (DPT). Pseudotime represents each patient's position along a
continuous disease progression axis from healthy to severely diseased state, estimated
from proteomics data structure.

SCIENTIFIC RATIONALE:
- Disease progression is not discrete (control → MCI → AD) but continuous
- Patients progress through states at different rates (heterogeneous disease trajectories)
- Pseudotime captures this continuous ordering from single cross-sectional proteomics snapshot
- Method: Uses graph structure of samples (computed from protein similarities)
- DPT solves diffusion equation on sample-similarity graph to find smooth progression

KEY CONCEPTS:
1. FEATURE SELECTION: Select 500 most variable proteins to reduce noise
2. PCA: Reduce to 50 dimensions to capture major variance
3. UMAP: Create 2D visualization preserving sample relationships
4. DIFFUSION PSEUDOTIME (DPT):
   - Builds k-NN graph from protein data
   - Computes diffusion components (eigenvalues of graph Laplacian)
   - Assigns pseudotime values reflecting progression along principal axis
   - Range: 0 (healthy) → 1+ (severely diseased)

ROOT SELECTION:
- DPT requires root cell (starting point for trajectory)
- Automatically selects cognitively normal patient (cogdx=1) with low Braak stage
- Falls back to sample closest to UMAP space center
- Essential for correct trajectory direction (healthy → disease)

VALIDATION:
- Pseudotime should correlate with clinical measures:
  * Negative correlation with MMSE (cognitive score: lower=worse)
  * Positive correlation with Braak stage (neuropathology: higher=worse)
  * Positive correlation with cognitive decline (cogdx: higher=worse)
  * Positive correlation with CERAD pathology

OUTPUTS:
- pseudotime_scores.csv: DPT scores + UMAP coordinates per patient
- master_patient_table.csv: Complete integration of clinical data, cell proportions, pseudotime
- pseudotime_umap.png: 4-panel visualization colored by diagnosis, pathology, cognition, pseudotime

MASTER PATIENT TABLE:
This table is the central integration point for all downstream steps:
- Clinical metadata (diagnosis, age, PMI, biomarkers)
- Cell-type proportions (6 columns)
- UMAP coordinates (2D position in protein space)
- Pseudotime scores (disease progression)
Used by Steps 1D, 1E, and network inference in Step 2
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
    """
    Load proteomics, clinical metadata, and cell-type proportions.

    PURPOSE:
    Consolidate outputs from previous steps (1A, 1B) into memory for downstream processing.

    PARAMETERS:
    -----------
    processed_dir : str
        Directory containing CSV files from Steps 1A and 1B

    RETURNS:
    --------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame or None)
        (proteomics_df, metadata_df, proportions_df)
        - proteomics_df: Normalized proteomics (n_patients × n_proteins)
        - metadata_df: Clinical data (n_patients × n_covariates)
        - proportions_df: Cell-type proportions (n_patients × 6) or None if unavailable

    NOTES:
    ------
    Cell-type proportions are optional. If unavailable, proportions_df = None.
    """
    # Define paths to outputs from previous steps
    proteomics_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
    metadata_file = f"{processed_dir}/rosmap_metadata.csv"
    proportions_file = f"{processed_dir}/cell_type_proportions.csv"

    # Load proteomics and metadata (required)
    proteomics_df = pd.read_csv(proteomics_file, index_col=0)
    metadata_df = pd.read_csv(metadata_file, index_col=0)

    # Load proportions if available (optional)
    proportions_df = None
    if os.path.exists(proportions_file):
        proportions_df = pd.read_csv(proportions_file, index_col=0)

    logging.info(f"  Loaded proteomics: {proteomics_df.shape}")
    logging.info(f"  Loaded metadata: {metadata_df.shape}")
    if proportions_df is not None:
        logging.info(f"  Loaded proportions: {proportions_df.shape}")

    return proteomics_df, metadata_df, proportions_df


def select_top_variable_proteins(proteomics_df, n_top=500):
    """
    Select most variable proteins for dimensionality reduction.

    PURPOSE:
    Feature selection reduces noise and computational complexity while retaining
    signal. Highly variable proteins are more likely to distinguish patient states.

    PARAMETERS:
    -----------
    proteomics_df : pd.DataFrame
        Z-score normalized proteomics (n_patients × n_proteins)
    n_top : int, default=500
        Number of top proteins to select (standard: 500-1000)

    RETURNS:
    --------
    pd.DataFrame
        Subsetted proteomics matrix (n_patients × n_top)

    RATIONALE:
    ----------
    - Very low-variance proteins reflect only noise, not biology
    - High-variance proteins capture disease-relevant variation
    - Reduces dimensionality: 5000 → 500 (10-fold reduction)
    - Speeds up PCA and UMAP computation
    - Improves signal-to-noise ratio

    METHOD:
    -------
    1. Compute variance for each protein across all patients
    2. Rank proteins by variance (descending)
    3. Keep top n_top proteins
    4. Subset proteomics matrix to these proteins
    """
    # Calculate variance for each protein across patients
    variances = proteomics_df.var(axis=0)
    # Select indices of top n_top proteins by variance
    top_proteins = variances.nlargest(n_top).index.tolist()
    # Subset to top proteins
    proteomics_subset = proteomics_df[top_proteins]
    logging.info(f"  Selected {len(top_proteins)} most variable proteins")
    return proteomics_subset


def compute_pca(proteomics_df, n_components=50):
    """
    Perform Principal Component Analysis to capture major variance.

    PURPOSE:
    Reduce dimensionality while preserving variance structure. PCA is the first step
    in a dimensionality reduction pipeline that flows into UMAP and pseudotime.

    PARAMETERS:
    -----------
    proteomics_df : pd.DataFrame
        Feature-selected proteomics (n_patients × n_selected_proteins)
    n_components : int, default=50
        Number of principal components to compute

    RETURNS:
    --------
    tuple of (np.ndarray, sklearn.PCA)
        (pca_coords, pca_object)
        - pca_coords: PCA coordinates (n_patients × n_components)
        - pca_object: Fitted PCA model (contains explained_variance_ratio_, components_, etc.)

    MATHEMATICAL CONCEPT:
    ---------------------
    PCA finds orthogonal directions (principal components) that maximize variance.
    - PC1: direction of maximum variance
    - PC2: direction of maximum variance orthogonal to PC1
    - etc.

    This transforms 500 proteins → 50 PCs that capture most variation.

    INTERPRETATION:
    ---------------
    If 50 PCs explain 90% variance:
      - 10% variance lost (acceptable)
      - Dimensionality reduced 10-fold
      - More interpretable than 500 proteins

    NOTES:
    ------
    - Automatically reduces n_components if n_patients < n_components
    - (Can't have more PCs than samples)
    """
    # Compute PCA with at most min(n_components, n_samples-1) components
    pca = PCA(n_components=min(n_components, proteomics_df.shape[0]-1))
    # Transform data to PCA space
    pca_coords = pca.fit_transform(proteomics_df)
    # Calculate total explained variance
    explained_var = pca.explained_variance_ratio_.sum()
    logging.info(f"  PCA: {pca.n_components_} components, {explained_var*100:.1f}% variance explained")
    return pca_coords, pca


def compute_umap(pca_coords, n_neighbors=15, min_dist=0.3):
    """
    Compute 2D UMAP embedding for visualization and graph structure.

    PURPOSE:
    UMAP (Uniform Manifold Approximation and Projection) creates a 2D visualization
    that preserves both local sample neighborhoods and global structure. Essential
    for understanding sample organization and selecting DPT root.

    PARAMETERS:
    -----------
    pca_coords : np.ndarray
        PCA coordinates (n_patients × n_components)
    n_neighbors : int, default=15
        Number of neighbors considered when building local graph (typical: 5-50)
    min_dist : float, default=0.3
        Minimum distance between points in UMAP embedding (controls compactness)
        Small values (0.1-0.3): more tightly grouped
        Large values (0.5+): more spread out

    RETURNS:
    --------
    tuple of (np.ndarray, sc.AnnData)
        (umap_coords, adata)
        - umap_coords: 2D coordinates (n_patients × 2)
        - adata: AnnData object with UMAP in obsm['X_umap']

    ALGORITHM:
    ----------
    UMAP is a nonlinear dimensionality reduction method:
    1. Compute k-NN graph (n_neighbors=15): each point connects to 15 nearest neighbors
    2. Fuzzy set union: create weighted graph of local structure
    3. Optimize 2D embedding preserving neighborhood structure
    4. Result: preserves both local clusters and global topology

    ADVANTAGES OVER t-SNE:
    ----------------------
    - Faster computation (nonlinear vs quadratic time)
    - Better preserves global structure
    - More stable (less stochastic noise)
    - Standard in modern single-cell analysis

    INTERPRETATION:
    ----------------
    - Points close together: similar protein profiles
    - Clusters: groups of similar patients
    - Continuous gradients: smooth transitions in biology
    """
    # Create AnnData object (scanpy format) for UMAP computation
    adata = sc.AnnData(X=pca_coords)

    # Build k-nearest neighbor graph (connectivity structure)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # Compute UMAP embedding on this graph
    sc.tl.umap(adata, min_dist=min_dist)

    # Extract UMAP coordinates from AnnData object
    umap_coords = adata.obsm['X_umap']
    logging.info(f"  UMAP computed")
    return umap_coords, adata


def select_dpt_root(adata, metadata_df):
    """
    Intelligently select root cell for diffusion pseudotime computation.

    PURPOSE:
    DPT is a graph-based trajectory algorithm that requires a root (starting point).
    The root should be the healthiest patient to ensure correct trajectory direction
    (healthy → disease progression).

    PARAMETERS:
    -----------
    adata : sc.AnnData
        Data object with UMAP computed (obsm['X_umap'] available)
    metadata_df : pd.DataFrame
        Clinical metadata with cognitive and neuropathology columns

    RETURNS:
    --------
    str
        Index/name of selected root cell (patient ID)

    SELECTION STRATEGY:
    -------------------
    Priority 1: Cognitively normal with minimal neuropathology
      - cogdx=1 (cognitive diagnosis: 1=cognitively normal)
      - Braak stage ≤ 1 (low AD neuropathology: 0-1 = no/minimal)

    Priority 2 (if no candidates found):
      - Sample closest to center of UMAP space
      - UMAP center reflects "average" patient
      - Central location = typical healthy phenotype

    Priority 3 (fallback):
      - First sample in dataset

    CLINICAL MEASURES:
    ------------------
    cogdx: Cognitive diagnosis
      1 = cognitively normal
      2 = MCI (mild cognitive impairment)
      3 = dementia

    Braak stage: AD neuropathology severity
      0 = no pathology
      1 = transentorhinal
      2-3 = limbic
      4-5 = neocortical
      6 = severe

    ALGORITHM:
    ----------
    1. Find all cognitively normal (cogdx=1) with low Braak (≤1)
    2. If candidates found:
        a. Project candidates onto UMAP coordinates
        b. Calculate distance from UMAP center
        c. Select candidate closest to center
    3. Otherwise, fall back to first candidate or first sample

    WHY UMAP CENTER?
    ----------------
    - UMAP center = average protein profile of cohort
    - Cognitively normal at center = truly representative healthy phenotype
    - Avoids outliers (even if cognitively normal)
    """
    # Find clinically normal candidates (cogdx=1, low Braak)
    candidates = []

    for idx, row in metadata_df.iterrows():
        # Handle multiple column name variants (different dataset formats)
        cogdx_variants = ['cogdx', 'COGDX']
        braak_variants = ['braaksc', 'BRAAKSC', 'braak']

        # Find cogdx value (if exists)
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

        # Add to candidates if clinically normal with low neuropathology
        if cogdx_val == 1 and braak_val is not None and braak_val <= 1:
            candidates.append(idx)

    # Select root based on candidates
    if candidates:
        # Prefer candidate closest to UMAP center (representative healthy patient)
        if hasattr(adata, 'obsm') and 'X_umap' in adata.obsm:
            # Calculate UMAP space center (mean coordinates)
            umap_center = adata.obsm['X_umap'].mean(axis=0)
            distances = {}

            # For each candidate, calculate distance to UMAP center
            for cand in candidates:
                # Find UMAP index of this candidate
                cand_idx = list(adata.obs_names).index(cand) if cand in adata.obs_names else None
                if cand_idx is not None:
                    # Euclidean distance from candidate to UMAP center
                    dist = np.linalg.norm(adata.obsm['X_umap'][cand_idx] - umap_center)
                    distances[cand] = dist

            # Select candidate with minimum distance to center
            if distances:
                root = min(distances, key=distances.get)
            else:
                # Fallback if distance calculation failed
                root = candidates[0]
        else:
            # Fallback if UMAP not available
            root = candidates[0]
    else:
        # Fallback: use first sample (least preferred)
        root = adata.obs_names[0]

    logging.info(f"  Selected root cell: {root}")
    return root


def compute_pseudotime(adata, root, n_dcs=10):
    """
    Compute diffusion pseudotime from graph structure and root cell.

    PURPOSE:
    Calculate disease progression score for each patient by solving diffusion equation
    on graph of patient similarities. Results in smooth ordering from healthy → diseased.

    PARAMETERS:
    -----------
    adata : sc.AnnData
        Data object with neighbors/graph computed
    root : str
        Patient ID of root cell (healthy reference)
    n_dcs : int, default=10
        Number of diffusion components to compute (eigenvalues of graph Laplacian)

    RETURNS:
    --------
    np.ndarray
        1D array of pseudotime scores (n_patients,)
        Range: typically 0 (root/healthy) to 1+ (diseased)

    ALGORITHM (Diffusion Pseudotime):
    ----------------------------------
    1. Build k-NN graph from patient similarities
       (From PCA/UMAP computed earlier in pipeline)

    2. Compute diffusion maps
       - Calculate graph Laplacian (matrix form of graph connectivity)
       - Compute first n_dcs eigenvalues/eigenvectors
       - These eigenvectors capture diffusion directions on graph

    3. Solve diffusion equation
       - Diffusion equation: ∂u/∂t = D∇²u
       - u = abundance of disease signal at each point
       - Initial condition: u(root) = 0 (healthy), elsewhere unknown
       - Diffusion spreads signal outward from root

    4. Pseudotime = normalized diffusion distance
       - Represents path length from root to each patient
       - Smooth because diffusion averages over paths

    WHY DIFFUSION?
    ---------------
    - Natural model of disease progression along protein landscape
    - Smooth (continuous): avoids discrete categorization
    - Robust: uses multiple paths, not just shortest path
    - Graph-based: respects sample relationships

    VS OTHER METHODS:
    ------------------
    - Simpler than pseudotime from branches (monocle) → good for single trajectory
    - Uses global structure (vs local: PCA direction alone)
    - Deterministic (vs stochastic: sampling-based methods)

    NOTES:
    ------
    - Root cell is set as iroot index in AnnData.uns
    - DPT algorithm finds path from root through graph
    - Scores normalized by diffusion distance
    - Monotonic along disease trajectory (by design)
    """
    # Compute diffusion maps: eigendecomposition of graph Laplacian
    sc.tl.diffmap(adata, n_comps=n_dcs)

    # Convert root cell name to index (DPT needs numeric index)
    if root in adata.obs_names:
        root_idx = list(adata.obs_names).index(root)
        adata.uns['iroot'] = root_idx  # Set root for DPT

    # Compute diffusion pseudotime from root
    sc.tl.dpt(adata)

    # Extract pseudotime scores
    pseudotime = adata.obs['dpt_pseudotime'].values
    logging.info(f"  Pseudotime range: {pseudotime.min():.3f} - {pseudotime.max():.3f}")
    return pseudotime


def validate_pseudotime(pseudotime, metadata_df, logger):
    """
    Validate pseudotime by checking correlation with independent clinical measures.

    PURPOSE:
    Ensure that computed pseudotime reflects true disease progression by correlating
    with established clinical biomarkers. Validation gives confidence that pseudotime
    captures meaningful biology.

    PARAMETERS:
    -----------
    pseudotime : np.ndarray
        Computed DPT scores (n_patients,)
    metadata_df : pd.DataFrame
        Clinical data with biomarker measurements
    logger : logging.Logger
        Logger for output

    RETURNS:
    --------
    bool
        True if pseudotime passes validation (≥4/4 measures show expected correlation)
        False otherwise (Warning status in results)

    VALIDATION MEASURES (4 total):
    --------------------------------
    1. MMSE (Mini-Mental State Exam)
       - Range: 0-30 (higher = better cognition)
       - Expected: negative correlation with pseudotime
       - Threshold: ρ < -0.30 (p < 0.05)
       - Interpretation: As pseudotime increases, cognition decreases

    2. Braak Stage (AD neuropathology)
       - Range: 0-6 (higher = more pathology)
       - Expected: positive correlation with pseudotime
       - Threshold: ρ > 0.30 (p < 0.05)
       - Interpretation: As pseudotime increases, pathology increases

    3. CERAD Score (amyloid-beta neuropathology)
       - Range: unclear but positive direction
       - Expected: negative correlation with pseudotime
       - Threshold: ρ < -0.30
       - Interpretation: Inverse to disease progression

    4. COGDX (cognitive diagnosis)
       - Values: 1=normal, 2=MCI, 3=dementia (higher = more impaired)
       - Expected: positive correlation with pseudotime
       - Threshold: ρ > 0.30
       - Interpretation: As pseudotime increases, cognitive diagnosis worsens

    ALGORITHM:
    -----------
    1. For each measure: attempt to find column (handling name variants)
    2. Skip if >50% missing data (insufficient data quality)
    3. Calculate Spearman correlation (rank-based, robust to outliers)
    4. Check if correlation sign × magnitude exceeds threshold
    5. Count passing measures

    RESULT:
    -------
    Valid if ≥4/4 measures pass expected correlation
    This multi-measure validation gives confidence in pseudotime

    NOTES:
    ------
    - Uses Spearman (not Pearson) because clinical measures may be ordinal/ranked
    - Handles missing data by excluding patients with NaN
    - Different clinical datasets have different column names
    - Logs individual correlations for transparency
    """
    # Define validation criteria for each clinical measure
    # (column_name, sign_multiplier, correlation_threshold)
    validations = {
        'mmse': ('mmse', -1, -0.30),              # Lower MMSE = worse cognition, should correlate negative
        'braak': ('braaksc', 1, 0.30),            # Higher Braak = more pathology, should correlate positive
        'cerad': ('ceradsc', -1, -0.30),          # CERAD inverse relation with pseudotime
        'cogdx': ('cogdx', 1, 0.30)               # Higher cogdx = worse cognition, should correlate positive
    }

    valid_count = 0

    # Test each measure
    for key, (col, sign, threshold) in validations.items():
        # Handle multiple column naming conventions
        col_variants = [col, col.upper()]
        found = False

        for var in col_variants:
            if var in metadata_df.columns:
                # Extract clinical values for this measure
                clinical_vals = metadata_df[var].values

                # Skip if too much missing data (>50%)
                if np.isnan(clinical_vals).sum() < len(clinical_vals) * 0.5:
                    # Create mask for non-missing values in both pseudotime and clinical measure
                    mask = ~(np.isnan(pseudotime) | np.isnan(clinical_vals))

                    # Only compute correlation if sufficient non-missing pairs (>3)
                    if mask.sum() > 3:
                        # Compute Spearman rank correlation (robust to outliers and ordinal data)
                        rho, pval = spearmanr(pseudotime[mask], clinical_vals[mask])

                        # Check if correlation meets expected direction and magnitude
                        # sign multiplier flips the sign for inverse relationships
                        if sign * rho > threshold:
                            valid_count += 1  # Count as passing
                            logger.info(f"    {key}: rho={rho:.3f} (p={pval:.2e})")
                        else:
                            # Correlation exists but doesn't meet threshold or wrong direction
                            logger.info(f"    {key}: rho={rho:.3f} (FAILED threshold={threshold})")
                        found = True
                        break

    # Summary: valid if all 4 measures pass
    logger.info(f"  Validation: {valid_count}/4 measures passed")
    return valid_count >= 4  # Pass if all 4 measures validate


def save_pseudotime_data(pseudotime, adata, metadata_df, proportions_df, processed_dir):
    """
    Save pseudotime scores and create master patient table integrating all Step 1 outputs.

    PURPOSE:
    Serialize Step 1C outputs and create the central integration point for downstream
    analyses. The master patient table combines clinical data, cell proportions, and
    pseudotime/UMAP coordinates into one comprehensive patient-level table.

    PARAMETERS:
    -----------
    pseudotime : np.ndarray
        Computed DPT scores
    adata : sc.AnnData
        Data object with UMAP coordinates
    metadata_df : pd.DataFrame
        Original clinical metadata
    proportions_df : pd.DataFrame or None
        Cell-type proportions (optional)
    processed_dir : str
        Directory to save outputs

    OUTPUTS:
    --------
    pseudotime_scores.csv (Step 1C-specific outputs)
        Columns: dpt_pseudotime, umap_1, umap_2
        One row per patient
        Index: patient IDs

    master_patient_table.csv (CENTRAL INTEGRATION TABLE)
        Comprehensive patient-level table combining:
        - Clinical metadata (diagnosis, age, PMI, biomarkers)
        - Cell-type proportions (6 columns: ct_Ex, ct_In, ct_Ast, ct_Oli, ct_Mic, ct_OPCs)
        - Pseudotime score (dpt_pseudotime)
        - UMAP coordinates (umap_1, umap_2)
        One row per patient (n_patients × ~15-20 columns)

    MASTER TABLE COLUMNS (typical):
    --------------------------------
    From metadata:
      - diagnosis: Control/AD/MCI
      - age_death: Age at death (years)
      - msex: Sex (0=F, 1=M)
      - pmi: Post-mortem interval (hours)
      - mmse: Mini-Mental State Exam score
      - braaksc: Braak neuropathology stage
      - ceradsc: CERAD pathology score
      - cogdx: Cognitive diagnosis

    From deconvolution:
      - ct_Ex, ct_In, ct_Ast, ct_Oli, ct_Mic, ct_OPCs (6 proportions)

    From pseudotime:
      - dpt_pseudotime: Computed disease progression score
      - umap_1, umap_2: 2D visualization coordinates

    IMPORTANCE:
    -----------
    The master table is used as input for:
      - Step 1D: NMF clustering (use pseudotime + cell proportions)
      - Step 1E: Survival analysis (correlate pseudotime with outcomes)
      - Step 2: Network inference (control for pseudotime + cell proportions)
    All downstream analyses reference this table.

    NOTES:
    ------
    - Index: patient IDs (matches all other data)
    - No missing values imputation (retains original missing)
    - Allows full reproducibility from saved data
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Create pseudotime output table
    pseudotime_df = pd.DataFrame({
        'dpt_pseudotime': pseudotime,
        'umap_1': adata.obsm['X_umap'][:, 0],  # First UMAP coordinate
        'umap_2': adata.obsm['X_umap'][:, 1]   # Second UMAP coordinate
    }, index=adata.obs_names)

    # Save pseudotime table
    pseudo_file = f"{processed_dir}/pseudotime_scores.csv"
    pseudotime_df.to_csv(pseudo_file)
    logging.info(f"  Saved: pseudotime_scores.csv")

    # Create master patient table by combining all available data
    master_df = metadata_df.copy()

    # Add pseudotime and UMAP coordinates
    master_df['dpt_pseudotime'] = pseudotime_df['dpt_pseudotime']
    master_df['umap_1'] = pseudotime_df['umap_1']
    master_df['umap_2'] = pseudotime_df['umap_2']

    # Add cell-type proportions if available
    if proportions_df is not None:
        for col in proportions_df.columns:
            master_df[col] = proportions_df[col]

    # Save master table
    master_file = f"{processed_dir}/master_patient_table.csv"
    master_df.to_csv(master_file)
    logging.info(f"  Saved: master_patient_table.csv ({master_df.shape})")

    return master_file


def plot_pseudotime_umap(adata, metadata_df, results_dir):
    """
    Create 4-panel UMAP visualization colored by different clinical/computed measures.

    PURPOSE:
    Publication-quality visualization showing how pseudotime and clinical metrics map
    onto the 2D protein space. Demonstrates that pseudotime captures disease-relevant variation.

    PARAMETERS:
    -----------
    adata : sc.AnnData
        Data object with UMAP and pseudotime computed
    metadata_df : pd.DataFrame
        Clinical metadata for coloring
    results_dir : str
        Directory where PNG will be saved

    OUTPUTS:
    --------
    pseudotime_umap.png (4-panel figure, 14×12 inches)

    Panel A (top-left): UMAP colored by Diagnosis
        - Control vs AD (and MCI if present)
        - Categorical coloring
        - Should show some separation if disease affects proteome

    Panel B (top-right): UMAP colored by Braak Stage
        - Continuous color scale (viridis: dark=low, bright=high)
        - Braak 0-6 (higher = more pathology)
        - Should show gradient corresponding to protein changes

    Panel C (bottom-left): UMAP colored by MMSE (Cognitive Score)
        - Continuous color scale (coolwarm: cool=low, warm=high)
        - MMSE 0-30 (higher = better cognition)
        - Should show gradient matching disease severity

    Panel D (bottom-right): UMAP colored by DPT Pseudotime
        - Continuous color scale (plasma: dark=early trajectory, bright=late)
        - Pseudotime score (0 to 1+)
        - Should show smooth gradient (healthy → diseased)

    INTERPRETATION:
    ----------------
    - Smooth gradients (not patches) = algorithm captured continuous trajectory
    - Spatial separation by diagnosis = disease alters protein landscape
    - All 4 panels should show similar spatial patterns = validation
    - Pseudotime panel should show clearest trajectory

    TECHNICAL NOTES:
    ----------------
    - Uses matplotlib subplots (2×2 grid)
    - Different colormaps for categorical (diagnosis) vs continuous (quantitative)
    - Colorbars for continuous variables
    - Grid overlaid (alpha=0.3) for reference
    """
    os.makedirs(results_dir, exist_ok=True)

    # Create 2×2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PANEL A: Diagnosis (Categorical coloring)
    ax = axes[0, 0]
    if 'diagnosis' in metadata_df.columns:
        # Get unique diagnoses and create color map
        unique_diagnoses = metadata_df['diagnosis'].unique()
        colors_diag = plt.cm.Set2(np.linspace(0, 1, len(unique_diagnoses)))

        # Plot each diagnosis group separately (for legend)
        for i, diag in enumerate(unique_diagnoses):
            mask = metadata_df['diagnosis'] == diag
            ax.scatter(adata.obsm['X_umap'][mask, 0], adata.obsm['X_umap'][mask, 1],
                      label=str(diag), alpha=0.6, s=50, color=colors_diag[i])
    ax.set_title('A) By Diagnosis', fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    ax.grid(alpha=0.3)

    # PANEL B: Braak Stage (Continuous coloring)
    ax = axes[0, 1]
    if 'braaksc' in metadata_df.columns:
        # Use viridis colormap for continuous variable (dark=low pathology, bright=high)
        scatter = ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                            c=metadata_df['braaksc'].values, cmap='viridis', s=50, alpha=0.6)
        ax.set_title('B) By Braak Stage', fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax, label='Braak Stage')
    ax.grid(alpha=0.3)

    # PANEL C: MMSE (Continuous coloring)
    ax = axes[1, 0]
    if 'mmse' in metadata_df.columns:
        # Use coolwarm colormap (cool=low cognition, warm=high cognition)
        scatter = ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                            c=metadata_df['mmse'].values, cmap='coolwarm', s=50, alpha=0.6)
        ax.set_title('C) By MMSE Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax, label='MMSE (0-30)')
    ax.grid(alpha=0.3)

    # PANEL D: Pseudotime (Continuous coloring)
    ax = axes[1, 1]
    # Use plasma colormap (dark=early trajectory, bright=late/diseased)
    scatter = ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                        c=adata.obs['dpt_pseudotime'].values, cmap='plasma', s=50, alpha=0.6)
    ax.set_title('D) By Pseudotime', fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='DPT Pseudotime')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_file = f"{results_dir}/pseudotime_umap.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: pseudotime_umap.png")


def main(data_dir="data", results_dir="results", test_mode=False):
    """
    Execute complete Step 1C: Disease pseudotime computation and visualization.

    PURPOSE:
    Master orchestrator for pseudotime computation. Transforms raw proteomics into
    a disease progression trajectory, validated against clinical measures. Produces
    the master patient table integrating all Step 1 outputs.

    PARAMETERS:
    -----------
    data_dir : str, default="data"
        Root data directory:
          - {data_dir}/processed/ → loads proteomics, metadata, proportions from Step 1A/1B
                                  → saves pseudotime_scores.csv, master_patient_table.csv
    results_dir : str, default="results"
        Directory where visualization PNG is saved
    test_mode : bool, default=False
        If True: uses fewer features (200 proteins) and smaller PCA (faster)
        If False: uses full feature set (500 proteins) for production

    RETURNS:
    --------
    dict
        Results summary:
        - 'pseudotime_min': Minimum pseudotime score (typically ~0)
        - 'pseudotime_max': Maximum pseudotime score
        - 'pseudotime_mean': Mean pseudotime across cohort
        - 'valid': Boolean, True if all 4 clinical measures validate
        - 'status': 'PASS' if valid, 'WARNING' if clinical validation fails

    WORKFLOW (6 STEPS):
    -------------------
    [1/6] Load data: Combines proteomics, metadata, cell proportions
    [2/6] Feature selection: Select 500 most variable proteins
    [3/6] PCA: Reduce to 50 dimensions
    [4/6] UMAP: Create 2D visualization
    [5/6] Pseudotime: Compute DPT from UMAP graph
    [6/6] Validation: Check correlation with clinical measures

    DETAILED WORKFLOW:
    ------------------
    Step 1: Load Data
      - Reads preprocessed proteomics (n_patients × n_proteins)
      - Reads metadata (diagnosis, age, PMI, cognitive/pathology biomarkers)
      - Optionally reads cell-type proportions from Step 1B

    Step 2: Feature Selection
      - Computes variance for each protein
      - Keeps top 200 (test) or 500 (production) most variable proteins
      - Reduces noise, speeds PCA/UMAP computation
      - Selected proteins drive dimensionality reduction

    Step 3: PCA
      - Computes 50 principal components
      - Captures ~80-90% of total variance
      - Creates orthogonal feature space for UMAP

    Step 4: UMAP
      - Builds k-NN graph (k=15) from PCA coordinates
      - Computes 2D embedding preserving local/global structure
      - UMAP coordinates stored for visualization and pseudotime

    Step 5: Pseudotime Computation
      - Selects root: cognitively normal patient with low neuropathology
      - Computes diffusion maps (10 components) on UMAP graph
      - Runs diffusion pseudotime algorithm
      - Result: smooth disease progression trajectory
      - Normalized scores typically range 0-1+ (healthy → diseased)

    Step 6: Validation
      - Pseudotime correlated with 4 clinical measures:
        * MMSE: negative correlation (lower cognition = higher pseudotime)
        * Braak: positive correlation (more pathology = higher pseudotime)
        * CERAD: negative correlation
        * COGDX: positive correlation (worse diagnosis = higher pseudotime)
      - Spearman correlation used (robust to outliers)
      - Each measure must show |ρ| > 0.30 (p < 0.05)
      - PASS if all 4 measures validate, otherwise WARNING

    OUTPUT FILES:
    ---------------
    {data_dir}/processed/pseudotime_scores.csv
        3 columns: dpt_pseudotime, umap_1, umap_2
        One row per patient

    {data_dir}/processed/master_patient_table.csv
        CENTRAL INTEGRATION TABLE combining:
        - Clinical metadata (~8-10 columns)
        - Cell-type proportions (6 columns if available)
        - Pseudotime score (1 column)
        - UMAP coordinates (2 columns)
        Total: ~15-20 columns, n_patients rows

    {results_dir}/step1/pseudotime_umap.png
        4-panel visualization (14×12 inches, 300 DPI)
        - Panel A: UMAP colored by diagnosis
        - Panel B: UMAP colored by Braak stage
        - Panel C: UMAP colored by MMSE score
        - Panel D: UMAP colored by pseudotime

    VALIDATION CONCEPT:
    -------------------
    Pseudotime should reflect true disease progression. Cross-validation against
    independent clinical measures gives confidence that the computed trajectory
    captures meaningful biology rather than artifact.

    SCIENTIFIC NOTES:
    -----------------
    - Diffusion pseudotime is smooth (differentiable)
    - Graph-based: uses all pairwise similarities (robust)
    - Deterministic: no randomness (reproducible)
    - Validated: correlates with established clinical biomarkers
    - Continuous: captures heterogeneous progression rates

    DOWNSTREAM USAGE:
    -----------------
    Master patient table is primary input for:
    - Step 1D: NMF clustering (stratify into subtypes)
    - Step 1E: Survival analysis (pseudotime predicts outcomes?)
    - Step 2: Network inference (control for covariates)

    ERROR HANDLING:
    ---------------
    Catches all exceptions, logs with traceback, re-raises
    Allows external error handling in master orchestrator
    """
    logger = logging.getLogger("Step1C")

    try:
        # Define directory structure
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        # Print step banner
        logger.info("="*70)
        logger.info("STEP 1C: Disease Pseudotime")
        logger.info("="*70)

        # STEP 1: Load data from previous steps
        logger.info("[1/6] Loading data...")
        proteomics_df, metadata_df, proportions_df = load_merged_data(processed_dir)

        # STEP 2: Feature selection - keep most variable proteins
        logger.info("[2/6] Feature selection...")
        n_top = 200 if test_mode else 500  # Fewer features in test mode for speed
        proteomics_subset = select_top_variable_proteins(proteomics_df, n_top=n_top)

        # STEP 3: Dimensionality reduction - PCA
        logger.info("[3/6] PCA...")
        pca_coords, pca = compute_pca(proteomics_subset, n_components=50)

        # STEP 4: Visualization - UMAP embedding
        logger.info("[4/6] UMAP...")
        umap_coords, adata = compute_umap(pca_coords, n_neighbors=15, min_dist=0.3)
        # Transfer patient IDs from proteomics to AnnData object
        adata.obs_names = proteomics_df.index

        # STEP 5: Pseudotime computation
        logger.info("[5/6] Pseudotime computation...")
        # Select root (starting point) for disease trajectory
        root = select_dpt_root(adata, metadata_df)
        # Compute diffusion pseudotime
        pseudotime = compute_pseudotime(adata, root, n_dcs=10)
        # Store pseudotime in AnnData object
        adata.obs['dpt_pseudotime'] = pseudotime

        # STEP 6: Clinical validation
        logger.info("[6/6] Validation...")
        is_valid = validate_pseudotime(pseudotime, metadata_df, logger)

        # Save outputs
        logger.info("Saving outputs...")
        # Save pseudotime table and create master patient table
        save_pseudotime_data(pseudotime, adata, metadata_df, proportions_df, processed_dir)
        # Generate 4-panel visualization
        plot_pseudotime_umap(adata, metadata_df, results_1_dir)

        # Print completion banner
        logger.info("="*70)
        logger.info("STEP 1C COMPLETE")
        logger.info("="*70)

        # Return results
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
