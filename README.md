# Alzheimer's Proteomics and Graph-Learning Pipeline

A staged Python research pipeline for processing proteomics, estimating cell-type composition, exploring disease-related trajectories, constructing networks and experimenting with graph neural networks.

**Status: research prototype.** Synthetic/test-mode outputs demonstrate workflow behavior; they do not establish independent clinical validation, causal drug targets or production readiness. Earlier stronger claims in the [historical README](docs/README_LEGACY.md) are not endorsed by this overview.

## For reviewers

Read [the employer guide](docs/EMPLOYER_GUIDE.md) for the technical story and a focused code tour.

| Stage | Implementation | Purpose |
|---|---|---|
| 1 | [run_step1.py](run_step1.py), [src/step1/](src/step1/) | QC, deconvolution, pseudotime, AsymAD/subtype exploration and validation |
| 2 | [run_step2.py](run_step2.py), [src/step2/](src/step2/) | WGCNA, graphical lasso, causal-method experiments and network consensus |
| 3 | [run_step3.py](run_step3.py), [src/step3/](src/step3/) | Feature engineering, graph model/training, explanations, stability and baselines |

The graph/causal terminology describes implemented analyses, not a demonstrated causal discovery result.

## Design

```text
Proteomics + clinical metadata + reference profiles
  -> quality control and adjusted features
  -> composition / trajectory / subtype analyses
  -> candidate networks
  -> graph-learning experiments and stability checks
```

## Setup and first inspection

The dependency file is **Colab-oriented**, not a complete standalone lockfile: it intentionally relies on Colab's existing NumPy/pandas/SciPy/scikit-learn/PyTorch stack. Use an isolated compatible scientific environment and read its comments before installing.

```bash
git clone https://github.com/SalmanSattar24/ad_gnn_pipeline.git
cd ad_gnn_pipeline
python -m pip install -r requirements.txt
python run_step1.py --help
python run_step2.py --help
python run_step3.py --help
```

After reviewing the configuration and computational requirements, a synthetic Step-1 exploration is exposed as `python run_step1.py --test`. This can still be computationally substantial; it is not a promise of a fast, fully independent clinical benchmark.

## Data and outputs

Review [config.yaml](config.yaml), [QUICK_START.md](QUICK_START.md) and [TEST_MODE_SIMPLIFICATIONS.md](TEST_MODE_SIMPLIFICATIONS.md). Real ROSMAP/Synapse inputs require separately authorized access and correct schema/path configuration. Keep tokens, participant-level data and generated private outputs out of Git.

The runners write intermediate tables, figures and logs for later stages. Synthetic correlations against generated clinical variables must not be presented as validation on independent patients.

## What this project demonstrates

- Modular, multi-stage scientific Python orchestration.
- Integration of omics preprocessing, statistical networks and graph-learning experiments.
- Explicit distinction between simplified test mode and data-backed research claims.
- Configuration, logging and intermediate-artifact organization.

## Limitations

No new full pipeline execution, clinical performance audit or model training was performed for this documentation refresh. Historical notebooks and examples may require environment/path adjustments. This is research software, not a diagnostic or treatment system. Preserve dataset, library and collaborator attribution when reusing it.
