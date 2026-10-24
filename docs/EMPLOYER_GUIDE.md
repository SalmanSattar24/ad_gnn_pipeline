# Employer guide

## The engineering question

How can a large proteomics investigation be divided into inspectable stages rather than one notebook with implicit state? The repository separates preprocessing/composition, network inference and graph learning, with intermediate artifacts and configurable runners.

## Suggested five-minute code tour

1. Read `run_step1.py`: runner responsibilities, logging and test-mode controls.
2. Inspect `src/step1/step_1a_load_preprocess.py` and `step_1c_pseudotime.py`.
3. Compare `src/step2/step_2b_glasso.py` with `step_2d_consensus.py`.
4. Inspect `src/step3/step_3b_model.py` and `step_3c_trainer.py`.
5. Read `TEST_MODE_SIMPLIFICATIONS.md` before interpreting any output.

## Evidence and boundaries

Source inspection supports the existence of these components. It does not establish that every real-data stage ran successfully or that a discovered graph edge is causal. Synthetic clinical associations are not independent clinical validation. No fresh training, restricted-data download or dependency installation is part of this documentation change.

## Discussable skills

Scientific pipeline decomposition, omics feature processing, graph representations, reproducibility and separating functional validation from scientific claims. For a résumé, describe the system and implemented analyses; do not claim clinically validated drug-target discovery.
