# Test Suite - Step 1 Pipeline

Test files for AD Research Pipeline Step 1.

## Available Tests

| Test | Covers | Command |
|------|--------|---------|
| `test_02_deconvolution.py` | Step 1B: Cell-Type Deconvolution | `python test_02_deconvolution.py` |
| `test_05_subtype_validation.py` | Step 1E: Subtype Validation | `python test_05_subtype_validation.py` |

## Running Individual Tests

From the root pipeline directory:

```bash
# Test deconvolution
python tests/test_02_deconvolution.py

# Test subtype validation
python tests/test_05_subtype_validation.py
```

## Running All Tests via Master Runner

The master runner (`run_step1.py`) orchestrates all steps including validation:

```bash
# Run all steps with test mode
python run_step1.py --test

# Run full pipeline
python run_step1.py
```

## Test Data

Tests use synthetic data generated on-the-fly:
- **Synthetic Mathys reference**: 8K cells × 10K genes × 6 cell types
- **Synthetic proteomics**: 180 samples × 5,000 proteins
- **Test mode**: 50 samples × 200 proteins (fast validation)

## Output Locations

Tests write output to:
- **Data files**: `../data/processed/`
- **Figures**: `../results/step1/`

---

**Note**: Tests 01, 03, and 04 are implemented as steps within the modular pipeline modules:
- Step 1A: `../src/step1/step_1a_load_preprocess.py`
- Step 1C: `../src/step1/step_1c_pseudotime.py`
- Step 1D: `../src/step1/step_1d_nmf_clustering.py`

Use the master runner (`run_step1.py`) to test all steps together.
