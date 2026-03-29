# Test-Mode Simplifications Log

> **⚠️ IMPORTANT:** All changes below were made **exclusively for pipeline correctness testing**.
> They MUST be reverted before running any real research analysis.
> These simplifications produce meaningless/toy results and are NOT scientifically valid.

---

## How to Switch Modes

In `notebooks/master_pipeline_runner.ipynb`, Cell 2:
```python
# For fast testing (~5 min):
TEST_MODE = True

# For real research runs:
TEST_MODE = False
```

---

## Simplification 1 — Synthetic Data Size
**File:** `src/step1/step_1a_load_preprocess.py`, line ~62

| Parameter | Test Value | **Research Value to Restore** |
|---|---|---|
| `n_patients` | `100` | `180` (or use real ROSMAP data) |
| `n_proteins` | `500` | `5000` (or use real ROSMAP data) |

```python
# Current (TEST):
def setup_synthetic_data(raw_data_dir, n_patients=100, n_proteins=500, test_mode=False):

# Restore to (RESEARCH):
def setup_synthetic_data(raw_data_dir, n_patients=180, n_proteins=5000, test_mode=False):
```

---

## Simplification 2 — Step 2A: WGCNA Co-expression
**File:** `src/step2/step_2a_wgcna.py`

### 2A-i: Cell-type restriction
```python
# Current (TEST): Only runs 1 cell type per subtype
if test_mode:
    cell_types = cell_types[:1]

# Restore to (RESEARCH): Remove this block entirely — run all cell types
```

### 2A-ii: Protein cap
```python
# Current (TEST): Only 10 proteins per cell type
if test_mode:
    ct_matrix = ct_matrix.iloc[:, :10]

# Restore to (RESEARCH): Remove this block — use all proteins from deconvolution
```

---

## Simplification 3 — Step 2B: Graphical Lasso (GLASSO)
**File:** `src/step2/step_2b_glasso.py`

### 2B-i: Cell-type restriction
```python
# Current (TEST): Only runs 1 cell type per subtype
if test_mode:
    cell_types = cell_types[:1]

# Restore to (RESEARCH): Remove this block — run all cell types
```

### 2B-ii: Protein cap
```python
# Current (TEST): Only 5 proteins per cell type
if test_mode:
    ct_matrix = ct_matrix.iloc[:, :5]

# Restore to (RESEARCH): Remove this block — the existing variance-filter (top 500) handles scaling
```

### 2B-iii: StARS subsamples
The `test_mode` branch in `run_glasso()` uses a fixed λ=0.5 with only `max_iter=50`.

```python
# Current (TEST): Fixed lambda, 50 iterations
gl = GraphicalLasso(alpha=0.5, max_iter=50)

# Restore to (RESEARCH):
# - N >= 70: GraphicalLassoCV(cv=5, max_iter=200)
# - 40-69:   StARS with n_subsamples=20 (already wired via non-test path)
```

---

## Simplification 4 — Step 2C: Causal Inference (GENIE3)
**File:** `src/step2/step_2c_causal.py`

### 2C-i: Cell-type restriction
```python
# Current (TEST): Only runs 1 cell type per subtype
if test_mode:
    cell_types = cell_types[:1]

# Restore to (RESEARCH): Remove this block — run all cell types
```

### 2C-ii: Protein cap
```python
# Current (TEST): Only 5 proteins per cell type
if test_mode:
    cols = ct_matrix.columns[:5]
    ct_matrix = ct_matrix[cols]

# Restore to (RESEARCH): Remove this block — the existing variance-filter (top 500) handles scaling
```

### 2C-iii: Random Forest estimator count
```python
# Current (TEST): 10 trees (meaningless feature importances)
n_estimators = 10 if test_mode else 100

# Restore to (RESEARCH): The else branch (100) already fires, no change needed.
# But review whether 100 is sufficient for real 500-protein GENIE3; literature suggests 500-1000.
```

> **Research note:** For publication-grade GENIE3, consider increasing to `n_estimators=500`.

---

## Simplification 5 — Step 3E: GNN Stability Protocol
**File:** `src/step3/step_3e_stability.py`

| Parameter | Test Value | **Research Value to Restore** |
|---|---|---|
| `n_seeds` | `2` | `30` (per research plan) |
| `epochs` | `3` | `50` |
| `top_k` (top proteins) | `5` | `20` |
| Stability threshold | `60% of 2 seeds = 1` | `60% of 30 = 18 runs` |

```python
# Current (TEST):
if test_mode:
    n_seeds = 2
    epochs = 3
    top_k = 5
else:
    epochs = 50
    top_k = 20

# Restore to (RESEARCH): The else branch restores the correct research values.
# Ensure TEST_MODE = False in the notebook.
```

---

## Summary Checklist for Research Runs

Before running any real research analysis, verify:

- [ ] `TEST_MODE = False` in `notebooks/master_pipeline_runner.ipynb`
- [ ] `n_patients=180`, `n_proteins=5000` in `step_1a_load_preprocess.py` (or use real ROSMAP data)
- [ ] WGCNA: all cell types processed, no 10-protein cap
- [ ] GLASSO: all cell types processed, no 5-protein cap, StARS with n_subsamples=20
- [ ] GENIE3: all cell types processed, no 5-protein cap, consider n_estimators=500
- [ ] Stability: 30 seeds, 50 epochs, top-20 proteins
- [ ] Real ROSMAP proteomics data loaded into `data/raw/` (replace synthetic data)

---

## Notes on Real Data Integration

When switching from synthetic to real ROSMAP data:
1. Place files in `data/raw/`:
   - `raw_proteomics.csv` — patients × proteins abundance matrix
   - `raw_metadata.csv` — clinical metadata (diagnosis, age, sex, PMI, braaksc, ceradsc, etc.)
2. The pipeline's `setup_synthetic_data()` in Step 1A will be **skipped automatically** if these files already exist.
3. Confirm WGCNA scale-free topology R² ≥ 0.85 — the synthetic data routinely defaults to β=6 fallback, which is expected and acceptable.
