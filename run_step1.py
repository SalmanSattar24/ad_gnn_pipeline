#!/usr/bin/env python
"""
Master Runner for Step 1: Patient Stratification and Subtype Discovery

Orchestrates all 5 substeps with proper logging, dependency checking,
error handling, and progress reporting.

Usage:
    python run_step1.py                          # Normal mode
    python run_step1.py --test                   # Test mode (50 patients, 200 proteins)
    python run_step1.py --skip-deconvolution     # Skip Step 1B
    python run_step1.py --n-subtypes 2           # Override k selection
"""

import sys
import os
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from step1.step_1a_load_preprocess import main as step_1a_main
from step1.step_1b_deconvolution import main as step_1b_main
from step1.step_1c_pseudotime import main as step_1c_main
from step1.step_1d_nmf_clustering import main as step_1d_main
from step1.step_1e_subtype_validation import main as step_1e_main


class Step1Runner:
    """Orchestrates Step 1 pipeline execution."""

    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.results_dir = args.results_dir
        self.test_mode = args.test
        self.skip_deconvolution = args.skip_deconvolution
        self.n_subtypes = args.n_subtypes

        self.start_time = None
        self.results = {}
        self.logger = None

        self._setup_logging()

    def _setup_logging(self):
        """Set up logging to both console and file."""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        log_file = f'logs/step1_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        # Create logger
        self.logger = logging.getLogger('Step1Runner')
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '  %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Configure other loggers
        for name in ['Step1A', 'Step1B', 'Step1C', 'Step1D', 'Step1E']:
            sublogger = logging.getLogger(name)
            sublogger.setLevel(logging.INFO)
            sublogger.addHandler(file_handler)
            sublogger.addHandler(console_handler)

        self.logger.info(f"Logging initialized: {log_file}")

    def check_dependencies(self):
        """Verify all required Python packages are installed."""
        self.logger.info("\n" + "="*70)
        self.logger.info("Checking dependencies...")
        self.logger.info("="*70)

        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy',
            'sklearn', 'scanpy', 'anndata', 'umap'
        ]

        missing = []

        for package_name in required_packages:
            try:
                __import__(package_name)
                self.logger.info(f"  [OK] {package_name}")
            except ImportError:
                self.logger.warning(f"  [MISSING] {package_name}")
                missing.append(package_name)

        if missing:
            self.logger.error("\nMissing packages detected!")
            self.logger.error("Install with:")
            self.logger.error(f"  pip install {' '.join(missing)}")
            return False

        self.logger.info("All dependencies satisfied.\n")
        return True

    def check_data_availability(self):
        """Verify input data is available."""
        self.logger.info("\n" + "="*70)
        self.logger.info("Checking data availability...")
        self.logger.info("="*70)

        raw_data_dir = f"{self.data_dir}/raw"

        # Data will be generated if missing (for synthetic testing)
        if not os.path.exists(raw_data_dir):
            self.logger.info(f"  Raw data directory will be created: {raw_data_dir}")
        else:
            files = os.listdir(raw_data_dir)
            self.logger.info(f"  Found {len(files)} files in {raw_data_dir}")

        self.logger.info("Data availability check complete.\n")
        return True

    def run_step(self, step_name, step_num, step_func):
        """Run a single step with error handling and timing."""
        self.logger.info("\n" + "="*70)
        self.logger.info(f"STEP {step_num}/5: {step_name}")
        self.logger.info("="*70)

        step_start = time.time()

        try:
            # Call step function with appropriate arguments
            if step_num == 1:
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    test_mode=self.test_mode
                )
            elif step_num == 2:
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    skip_deconvolution=self.skip_deconvolution,
                    test_mode=self.test_mode
                )
            elif step_num == 4:
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    n_subtypes=self.n_subtypes,
                    test_mode=self.test_mode
                )
            else:
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    test_mode=self.test_mode
                )

            step_time = time.time() - step_start

            self.logger.info(f"\n[PASS] {step_name} completed in {step_time:.1f}s")
            self.results[f'step{step_num}'] = {
                'status': 'PASS',
                'time': step_time,
                'result': result
            }

            return True

        except Exception as e:
            step_time = time.time() - step_start
            self.logger.error(f"\n[FAIL] {step_name} failed after {step_time:.1f}s")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(traceback.format_exc())

            self.results[f'step{step_num}'] = {
                'status': 'FAIL',
                'time': step_time,
                'error': str(e)
            }

            return False

    def run_all(self):
        """Run all steps in sequence."""
        self.start_time = time.time()

        self.logger.info("\n\n")
        self.logger.info("#"*70)
        self.logger.info("# Step 1: Patient Stratification & Subtype Discovery")
        self.logger.info("#"*70)

        if self.test_mode:
            self.logger.info("TEST MODE: Subsampling to 50 patients, 200 proteins")
        else:
            self.logger.info("NORMAL MODE: Using full dataset")

        if self.skip_deconvolution:
            self.logger.info("DECONVOLUTION: Skipping (using equal cell-type proportions)")

        # Check dependencies
        if not self.check_dependencies():
            self.logger.error("Dependency check failed. Exiting.")
            return False

        # Check data
        if not self.check_data_availability():
            self.logger.error("Data check failed. Exiting.")
            return False

        # Run steps
        steps = [
            ("1A: Data Loading & Preprocessing", 1, step_1a_main),
            ("1B: Cell-Type Deconvolution", 2, step_1b_main),
            ("1C: Disease Pseudotime", 3, step_1c_main),
            ("1D: NMF Consensus Clustering", 4, step_1d_main),
            ("1E: Subtype Validation", 5, step_1e_main),
        ]

        success_count = 0
        for step_name, step_num, step_func in steps:
            if self.run_step(step_name, step_num, step_func):
                success_count += 1
            else:
                # On failure, stop execution
                self.logger.error(f"\nPipeline stopped due to failure in {step_name}")
                return False

        # Generate final report
        total_time = time.time() - self.start_time
        self._generate_final_report(success_count, total_time)

        return success_count == 5

    def _generate_final_report(self, success_count, total_time):
        """Generate final summary report."""
        self.logger.info("\n\n")
        self.logger.info("="*70)
        self.logger.info("FINAL SUMMARY REPORT")
        self.logger.info("="*70)

        # Status line
        status = "SUCCESS" if success_count == 5 else f"PARTIAL ({success_count}/5)"
        self.logger.info(f"\nOverall Status: {status}")
        self.logger.info(f"Total Runtime: {total_time/60:.1f} minutes ({total_time:.0f}s)")

        # Step results
        self.logger.info("\nStep Results:")
        for i in range(1, 6):
            step_key = f'step{i}'
            if step_key in self.results:
                result = self.results[step_key]
                status_str = result['status']
                time_str = f"{result['time']:.1f}s"
                self.logger.info(f"  Step 1{chr(64+i)}: {status_str:4s} ({time_str})")

                if status_str == 'PASS' and 'result' in result:
                    result_data = result['result']
                    # Print key metrics
                    if i == 1:
                        self.logger.info(f"        -> {result_data.get('n_samples')} samples x "
                                       f"{result_data.get('n_proteins')} proteins")
                    elif i == 3:
                        self.logger.info(f"        -> Pseudotime range: {result_data.get('pseudotime_min'):.3f} - "
                                       f"{result_data.get('pseudotime_max'):.3f}")
                    elif i == 4:
                        sizes = result_data.get('subtype_sizes', {})
                        sizes_str = ', '.join([f"{k}={v}" for k, v in sorted(sizes.items())])
                        self.logger.info(f"        -> k={result_data.get('n_subtypes')}: {sizes_str}")

        # Output locations
        self.logger.info(f"\nOutputs saved to:")
        self.logger.info(f"  Data: {os.path.abspath(self.data_dir)}/processed/")
        self.logger.info(f"  Figures: {os.path.abspath(self.results_dir)}/step1/")
        self.logger.info(f"  Logs: {os.path.abspath('logs')}/")

        self.logger.info("\n" + "="*70)
        self.logger.info("Documentation: See README.md for running individual steps")
        self.logger.info("="*70 + "\n")


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description='Master runner for AD Research Pipeline Step 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_step1.py                          # Normal mode
  python run_step1.py --test                   # Test mode (50 samples, 200 proteins)
  python run_step1.py --skip-deconvolution     # Skip Step 1B
  python run_step1.py --n-subtypes 3           # Force k=3 for clustering

Files:
  config.yaml          - Pipeline configuration and parameters
  logs/step1_run.log   - Execution log (new file per run)
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (50 patients, 200 proteins for rapid validation)'
    )

    parser.add_argument(
        '--skip-deconvolution',
        action='store_true',
        help='Skip Step 1B deconvolution (use equal cell-type proportions instead)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data/)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Path to results directory (default: results/)'
    )

    parser.add_argument(
        '--n-subtypes',
        type=int,
        default=None,
        help='Override automatic k selection (e.g., 2, 3, 4)'
    )

    args = parser.parse_args()

    # Run pipeline
    runner = Step1Runner(args)
    success = runner.run_all()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
