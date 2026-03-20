"""
Step 1: Patient Stratification and Subtype Discovery

OVERVIEW:
This step implements the complete data preprocessing and patient stratification pipeline
for AD research. It transforms raw proteomics data into patient subtypes with distinct
biological and clinical characteristics.

PIPELINE STRUCTURE (5 substeps):

1A: Data Loading & Preprocessing
    Input: Raw proteomics (n_patients × n_proteins), clinical metadata
    Process: QC filtering, KNN imputation, log2 transform, z-score norm, covariate regression
    Output: Clean proteomics, processed metadata
    Purpose: Remove technical noise, stabilize variance, remove batch effects

1B: Cell-Type Deconvolution
    Input: Bulk proteomics, snRNA-seq reference (Mathys 2019)
    Process: NNLS deconvolution (proportions of 6 cell types)
    Output: Cell-type proportions (n_patients × 6)
    Purpose: Reveal cellular composition of tissue samples

1C: Disease Pseudotime Computation
    Input: Proteomics, metadata, cell proportions
    Process: Feature selection → PCA → UMAP → Diffusion pseudotime
    Output: DPT scores, UMAP coordinates, master patient table
    Purpose: Create continuous disease progression trajectory

1D: NMF Consensus Clustering
    Input: Proteomics (AD/MCI patients only), master patient table
    Process: NMF consensus clustering (k=2-5, cophenetic validation)
    Output: Subtype labels (ST1, ST2, ...), master table with subtypes
    Purpose: Stratify AD/MCI into distinct disease subtypes

1E: Subtype Validation & Clinical Interpretation
    Input: Master table with subtypes
    Process: Clinical association tests, cell-type labeling, visualizations
    Output: Survival curves, publication figures, summary report
    Purpose: Validate subtypes are clinically meaningful and interpretable

KEY DATA FLOW:
Step 1A (proteomics cleaning)
    ↓
Step 1B (cell-type proportions)
    ↓
Step 1C (pseudotime trajectory) → Creates master_patient_table
    ↓
Step 1D (subtype discovery) → Updates master_patient_table with subtypes
    ↓
Step 1E (subtype validation) → Publication figures

MASTER PATIENT TABLE (Central Integration Point):
After Step 1E, contains:
- Clinical metadata: diagnosis, age, PMI, biomarkers (MMSE, Braak, CERAD)
- Cell-type proportions: 6 columns (ct_Ex, ct_In, ct_Ast, ct_Oli, ct_Mic, ct_OPCs)
- Pseudotime score: dpt_pseudotime (disease progression)
- UMAP coordinates: umap_1, umap_2 (protein space visualization)
- Subtype label: subtype (ST1, ST2, ..., or Control)

Used as input for:
- Step 2: Gene regulatory networks (stratified by subtype)
- Step 3: Graph neural networks (subtype-specific mechanisms)
- Step 4: Mendelian randomization (subtype validation)
- Downstream: Survival analysis, drug response prediction

SUBMODULES:
  - step_1a_load_preprocess: Data loading & preprocessing (344 lines)
  - step_1b_deconvolution: Cell-type deconvolution (252 lines)
  - step_1c_pseudotime: Disease pseudotime computation (322 lines)
  - step_1d_nmf_clustering: NMF consensus clustering (297 lines)
  - step_1e_subtype_validation: Subtype validation & clinical mapping (381 lines)

TOTAL: ~1600 lines of production-ready code with comprehensive validation

EXECUTION:
All steps are orchestrated by run_step1.py in the parent directory.
Run individual steps with: step_1a.main(), step_1b.main(), etc.
Or run complete pipeline: python run_step1.py
"""

__version__ = "1.0.0"
__author__ = "AD Research Team"
