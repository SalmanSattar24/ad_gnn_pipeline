# Test-Mode Simplifications — Full Record

> **⚠️ CRITICAL:** Every change listed here was made **solely to validate pipeline
> plumbing and code correctness**. These modifications produce **scientifically meaningless
> toy results** and **MUST be reverted** before any real research analysis.
>
> This file was compiled from the full development history of this session.

---

## Master Switch

**File:** `notebooks/master_pipeline_runner.ipynb`, Cell 2

```python
TEST_MODE = True   # ← current state (fast testing)
TEST_MODE = False  # ← restore this for research
```

All pipeline stages read this flag through their `test_mode=` parameter.

---

## ARCHITECTURAL CHANGE: Step 2C Engine Swap

> **⚠️ Most significant change — requires special attention before research runs.**

### What was changed
The original causal inference engine (`arboreto` / `GRNBoost2` / Dask distributed) was
**completely replaced** with a custom pure scikit-learn GENIE3 equivalent using
`joblib.Parallel` + `RandomForestRegressor`.

**File:** `src/step2/step_2c_causal.py`

| | Original (research) | Current (test workaround) |
|---|---|---|
| **Engine** | `arboreto.algo.grnboost2` + Dask | Custom `RandomForestRegressor` loop |
| **Parallelism** | Dask distributed cluster | `joblib.Parallel(n_jobs=-1)` |
| **Algorithm** | GBM (Gradient Boosted Machines) | Random Forest |
| **Publication reference** | Huynh-Thu et al. 2010 (GENIE3), Moerman et al. 2019 (GRNBoost2) | N/A — test substitute |
| **Column names** | `TF`, `target`, `importance` | `protein_A`, `protein_B`, `weight` |

**Why it was changed:** Arboreto's Dask workers were OOM-killed by Google Colab's Docker
environment, causing `TypeError: Must supply at least one delayed object` regardless of
cluster configuration.

### To revert for research
```python
# Restore in src/step2/step_2c_causal.py:
from dask.distributed import Client, LocalCluster
from arboreto.algo import grnboost2

# Replace the joblib RF loop with:
network = grnboost2(expression_data=df_matrix,
                    tf_names=proteins,
                    client_or_address=client)

# Restore column rename:
edges.rename(columns={'TF': 'protein_A', 'target': 'protein_B', 'importance': 'weight'},
             inplace=True)
```

> **Research note:** If Colab continues to OOM-kill Dask workers, consider running Step 2C
> on a local machine or HPC cluster with adequate RAM (≥32 GB recommended for 500 proteins).

---

## Dependency Pinning (ABI Compatibility)

**File:** `requirements.txt`

Pinned to pre-NumPy 2.0 versions to fix `ValueError: numpy.dtype size changed` binary
incompatibility between NumPy 2.x and packages compiled against NumPy 1.x.

| Package | Pinned (test) | Target for research |
|---|---|---|
| `pandas` | `<=2.1.4` | Latest stable compatible with numpy |
| `numpy` | `<2.0.0` | `>=2.0` when all deps upgraded |
| `scipy` | `<=1.12.0` | Latest stable |
| `scikit-learn` | `<=1.4.2` | Latest stable |

> **Research note:** Before upgrading these pins, verify that `PyWGCNA`, `scanpy`,
> and `torch-geometric` all support NumPy 2.x. Check release notes for each.

---

## Step 1A — Data Loading & Preprocessing

**File:** `src/step1/step_1a_load_preprocess.py`

### 1A-i: Synthetic dataset size

```python
# Current (TEST):
def setup_synthetic_data(raw_data_dir, n_patients=100, n_proteins=500, ...):

# Restore (RESEARCH):
def setup_synthetic_data(raw_data_dir, n_patients=180, n_proteins=5000, ...):
```

**Impact:** 100×500 = 50K cells vs 180×5000 = 900K cells. The full 5000-protein matrix
is required for meaningful variance-based protein selection in Step 2.

> **Research note:** For actual ROSMAP data, delete synthetic defaults entirely and load
> real `raw_proteomics.csv` and `raw_metadata.csv` from the data access portal.

---

## Step 1B — Cell-Type Deconvolution

**File:** `src/step1/step_1b_deconvolution.py`

### 1B-i: Synthetic snRNA-seq reference size

The `generate_synthetic_mathys_reference()` function generates a scaled-down reference
when `test_mode=True`:

| Parameter | Test (test_mode=True) | Research (test_mode=False) |
|---|---|---|
| `n_cells` | 1,000 cells | 8,066 cells (matches Mathys 2019) |
| `n_genes` | 500 genes | 10,000 genes |
| `n_subjects` | 10 subjects | 48 subjects |

```python
# In generate_synthetic_mathys_reference():
n_cells = 1000 if test_mode else 8066
n_genes = 500 if test_mode else 10000
n_subjects = 10 if test_mode else 48
```

> **Research note:** For real research, replace the synthetic reference entirely with the
> real **Mathys et al. 2019** snRNA-seq dataset from Synapse (syn18681734).
> Download `mathys_reference.h5ad` and place in `data/raw/`.

---

## Step 2A — WGCNA Co-expression

**File:** `src/step2/step_2a_wgcna.py`

### 2A-i: Cell-type restriction (added this session)
```python
# Current (TEST): Only runs WGCNA on 1 cell type per subtype
if test_mode:
    cell_types = cell_types[:1]

# Restore (RESEARCH): Remove this block — process all 6 cell types
```

### 2A-ii: Protein cap per cell type (added this session)
```python
# Current (TEST): Cap at 10 proteins
if test_mode:
    ct_matrix = ct_matrix.iloc[:, :10]

# Restore (RESEARCH): Remove this block — use all proteins from deconvolution
```

### 2A-iii: Soft threshold selection
```python
# In compute_tom():
if test_mode:
    beta = 4  # Fixed — skips scale-free topology fitting entirely

# Restore (RESEARCH): Remove the test_mode branch — let pick_soft_threshold() run
# Research target: beta that achieves R² ≥ 0.85 scale-free topology fit
```

**Impact of restoring:** WGCNA will run on all 6 cell types × 2 subtypes = 12 networks,
each computing a full TOM matrix across all deconvolved proteins. Expected runtime:
~30–60 min on Colab GPU.

---

## Step 2B — Graphical Lasso (GLASSO)

**File:** `src/step2/step_2b_glasso.py`

### 2B-i: Cell-type restriction (added this session)
```python
# Current (TEST): Only runs GLASSO on 1 cell type per subtype
if test_mode:
    cell_types = cell_types[:1]

# Restore (RESEARCH): Remove this block — process all 6 cell types
```

### 2B-ii: Protein cap (added this session)
```python
# Current (TEST): Cap at 5 proteins (meaningless precision matrix)
if test_mode:
    ct_matrix = ct_matrix.iloc[:, :5]

# Restore (RESEARCH): Remove this block
# The existing variance-filter (top 500 proteins) handles scaling:
if not test_mode and ct_matrix.shape[1] > 500:
    ct_matrix = ct_matrix[variances.index[:500]]
```

### 2B-iii: Fixed lambda vs. StARS
```python
# Current (TEST): Fixed alpha=0.5, max_iter=50 — no bootstrapping at all
if test_mode:
    gl = GraphicalLasso(alpha=0.5, max_iter=50)

# Restore (RESEARCH): The non-test branches handle this correctly:
# N >= 70 → GraphicalLassoCV(cv=5, max_iter=200)
# 40–69  → StARS with n_subsamples=20, compute_stars_lambda()
```

### 2B-iv: Minimum sample size gate
```python
# Current code (applies always — keep in research too):
if n_patients < 40 and not test_mode:
    logger.info(f"Skipping GLASSO: N={n_patients} is < 40.")
    return pd.DataFrame(...)

# Research note: ST2 had N=23 patients, so GLASSO was legitimately skipped for ST2.
# This is scientifically correct behaviour — do NOT remove this gate.
```

**Impact of restoring:** StARS runs 20 bootstrap subsamples × 10 lambda values × 12
cell-type combos, each fitting a 500×500 precision matrix. Expected runtime: ~60–90 min.

---

## Step 2C — Causal Inference (GENIE3 / GRNBoost2)

**File:** `src/step2/step_2c_causal.py`

### 2C-i: Cell-type restriction (added this session)
```python
# Current (TEST): Only 1 cell type per subtype
if test_mode:
    cell_types = cell_types[:1]

# Restore (RESEARCH): Remove — run all 6 cell types
```

### 2C-ii: Protein cap (added this session)
```python
# Current (TEST): Cap at 5 proteins
if test_mode:
    cols = ct_matrix.columns[:5]
    ct_matrix = ct_matrix[cols]

# Restore (RESEARCH): Remove — existing variance-filter (top 500) handles scaling
```

### 2C-iii: Random Forest tree count
```python
# Current: 10 trees in test_mode, 100 in normal mode
n_estimators = 10 if test_mode else 100

# Restore (RESEARCH): The else branch already fires correctly (100 trees).
# For publication-grade results, consider increasing to 500–1000 trees.
# Literature (Huynh-Thu 2010) recommends 1000 trees for stable importance scores.
n_estimators = 1000  # publication target
```

### 2C-iv: Engine (see Architecture Change section above)
The entire Arboreto/GRNBoost2 engine was replaced with a scikit-learn RF loop.
See the "ARCHITECTURAL CHANGE" section at the top of this file for restoration steps.

---

## Step 2D — Consensus Network

**File:** `src/step2/step_2d_consensus.py`

### 2D-i: STRING API integration skipped in test_mode
```python
# Current (TEST): STRING API is skipped when test_mode=True
if not test_mode:
    string_bonus = fetch_string_edges(list(all_proteins))

# This is already correct behaviour — test_mode correctly skips the external API call.
# For research: ensure test_mode=False so STRING high-confidence edges are fetched.
```

**Impact of restoring:** STRING API will be queried for each (subtype, cell_type) pair.
Requires internet access from the compute environment. Uses required_score=700
(high-confidence interactions only).

---

## Step 3E — Stability Protocol

**File:** `src/step3/step_3e_stability.py`

| Parameter | Test value | Research value |
|---|---|---|
| `n_seeds` | **2** | **30** (per research plan) |
| `epochs` | **3** | **50** |
| `top_k` (top proteins tracked) | **5** | **20** |
| Stability threshold | 60% × 2 = **≥1 run** | 60% × 30 = **≥18 runs** |
| Batch size | 8 | 8 (unchanged — OK) |
| Learning rate | 0.001 | 0.001 (unchanged — OK) |

```python
# Current (TEST):
if test_mode:
    n_seeds = 2
    epochs = 3
    top_k = 5
else:
    epochs = 50
    top_k = 20

# Restore (RESEARCH): The else branch already gives correct research values.
# Ensure TEST_MODE = False in the notebook.
```

**Impact of restoring:** 30 seeds × 50 epochs × all subtype/cell-type data objects.
With GPU (T4 on Colab), expected runtime ~30–60 min for Step 3E alone.

---

## Step 1B — Deconvolution Method Note

When `skip_deconvolution=True` is passed (not currently set but exists in codebase):
- Cell-type proportions are set to uniform `1/6` for each cell type
- This completely bypasses NNLS and produces biologically meaningless proportions
- **Ensure this flag is False for all research runs**

---

## Summary Checklist — Before Any Research Run

Work through this list top to bottom before running the pipeline on real data:

**Environment:**
- [ ] `TEST_MODE = False` in `notebooks/master_pipeline_runner.ipynb`
- [ ] Remove pinned versions from `requirements.txt` if ecosystem has matured to NumPy 2.x

**Data:**
- [ ] Real ROSMAP proteomics placed at `data/raw/raw_proteomics.csv`
- [ ] Real ROSMAP metadata placed at `data/raw/raw_metadata.csv`
- [ ] Real Mathys 2019 snRNA-seq reference at `data/raw/mathys_reference.h5ad`
- [ ] Delete any pre-generated synthetic data in `data/raw/` and `data/processed/`

**Step 1A:**
- [ ] `n_patients=180`, `n_proteins=5000` restored in `setup_synthetic_data()` defaults
  (or confirm real data is loaded and synthetic generation is skipped)

**Step 2A (WGCNA):**
- [ ] Removed `cell_types = cell_types[:1]` test restriction
- [ ] Removed `ct_matrix.iloc[:, :10]` protein cap
- [ ] Test-mode fixed `beta=4` removed — `pick_soft_threshold()` runs freely

**Step 2B (GLASSO):**
- [ ] Removed `cell_types = cell_types[:1]` test restriction
- [ ] Removed `ct_matrix.iloc[:, :5]` protein cap
- [ ] Test-mode fixed `alpha=0.5` branch not triggered (`TEST_MODE=False`)
- [ ] StARS bootstrapping active (`n_subsamples=20`)

**Step 2C (GENIE3 / Causal):**
- [ ] Removed `cell_types = cell_types[:1]` test restriction
- [ ] Removed `ct_matrix.iloc[:, :5]` protein cap
- [ ] `n_estimators` set to 500–1000 for publication quality
- [ ] **CRITICAL:** Restore arboreto/GRNBoost2 engine OR confirm RF-GENIE3 is acceptable
  for publication (cite GENIE3 paper if keeping scikit-learn version)

**Step 2D (Consensus):**
- [ ] `TEST_MODE=False` ensures STRING API is queried
- [ ] Verify internet access is available from compute environment

**Step 3E (Stability):**
- [ ] `TEST_MODE=False` triggers 30 seeds, 50 epochs, top-20 proteins
- [ ] GPU confirmed active (`Using device: cuda`)

---

## Runtime Estimates (Research Mode, Colab T4 GPU)

| Stage | Estimated Time |
|---|---|
| Step 1 (all substeps) | ~10 min |
| Step 2A — WGCNA (12 networks) | ~30–60 min |
| Step 2B — GLASSO StARS (6 ST1 networks) | ~60–90 min |
| Step 2C — GENIE3 (12 networks × 500 proteins) | ~2–4 hours |
| Step 2D — Consensus | ~5 min |
| Step 3A — Feature Engineering | ~5 min |
| Step 3E — Stability (30 seeds × 50 epochs) | ~30–60 min |
| **Total** | **~5–8 hours** |

> Consider splitting into separate Colab sessions (Step 1 → save, Step 2 → save, Step 3)
> to avoid session timeouts. Each step saves outputs to disk before the next begins.
