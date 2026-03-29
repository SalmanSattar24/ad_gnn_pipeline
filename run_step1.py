#!/usr/bin/env python
"""
Master Runner for Step 1: Patient Stratification and Subtype Discovery

This module orchestrates the complete Step 1 pipeline, which consists of 5 sequential substeps:
  - Step 1A: Data loading, quality control, and preprocessing
  - Step 1B: Cell-type deconvolution (estimates proportion of 6 brain cell types per patient)
  - Step 1C: Disease pseudotime calculation (computes continuous disease progression score)
  - Step 1D: NMF consensus clustering (discovers disease subtypes from proteomics data)
  - Step 1E: Subtype validation (tests subtypes against clinical biomarkers)

The runner provides:
  1. Dependency checking (verifies required packages are installed)
  2. Logging to console and file (timestamped, for troubleshooting)
  3. Error handling and graceful failure (stops on error, captures full stack trace)
  4. Progress tracking (reports time per step, overall status)
  5. Argument parsing (supports flexible execution modes)

Usage:
    python run_step1.py                          # Normal mode (uses full dataset)
    python run_step1.py --test                   # Test mode (50 patients, 200 proteins, ~30s)
    python run_step1.py --skip-deconvolution     # Skip Step 1B (no snRNA-seq reference needed)
    python run_step1.py --n-subtypes 2           # Override automatic k selection

Note: All 5 steps are designed to be importable and runnable independently via their main() functions.
See STEP1_README.md or INDEX.md for more details.
"""

import sys
import os
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import traceback

# Add src directory to Python path so we can import step modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the main functions from each step module
# Each step has a main(data_dir, results_dir, test_mode, ...) function that handles execution
from step1.step_1a_load_preprocess import main as step_1a_main
from step1.step_1b_deconvolution import main as step_1b_main
from step1.step_1c_pseudotime import main as step_1c_main
from step1.step_1d_asymad import main as step_1d_main
from step1.step_1e_nmf_clustering import main as step_1e_main
from step1.step_1f_subtype_validation import main as step_1f_main


class Step1Runner:
    """
    Orchestrates sequential execution of all Step 1 substeps.

    This class manages:
    - Logging setup (console + file output with timestamps)
    - Dependency verification (ensures all required packages are installed)
    - Data availability checking (verifies input data or prepares synthetic data)
    - Step execution with error handling (runs steps sequentially, stops on failure)
    - Performance tracking (times each step, reports final summary)
    - Result aggregation (collects outputs from all steps for final report)

    Attributes:
        args: Command-line arguments (test, skip_deconvolution, n_subtypes, data_dir, results_dir)
        data_dir: Path to data folder (containing raw/ and processed/ subdirectories)
        results_dir: Path to results folder (where PNG figures are saved)
        test_mode: Boolean, if True uses synthetic data (50 samples x 200 proteins for speed)
        skip_deconvolution: Boolean, if True skips Step 1B (uses equal cell-type proportions)
        n_subtypes: Optional integer, overrides automatic k selection in Step 1D
        logger: Python logging.Logger instance for console + file output
        results: Dictionary storing status and metadata from each executed step
        start_time: Timestamp when pipeline started (for timing total execution)
    """

    def __init__(self, data_dir='data', results_dir='results', test_mode=False, 
                 skip_deconvolution=False, n_subtypes=None):
        """
        Initialize the runner with parameters and set up logging.
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.test_mode = test_mode
        self.skip_deconvolution = skip_deconvolution
        self.n_subtypes = n_subtypes

        # Initialize containers for execution tracking
        self.start_time = None  # Will be set when run_all() is called
        self.results = {}  # Will store {step1: {status, time, result}, ...}
        self.logger = None  # Will be initialized by _setup_logging()

        self._setup_logging()

    def _setup_logging(self):
        """
        Configure logging to output to both console and file.

        This method creates:
        1. A file handler that writes to logs/step1_run_YYYYMMDD_HHMMSS.log
           - Includes full timestamp, logger name, level, and message
           - Used for debugging and audit trails
        2. A console handler that writes to stdout
           - Uses minimal formatting for user-friendly output
           - Duplicates file output for real-time monitoring
        3. Loggers for the main runner and all 5 substeps
           - All loggers write to same file and console handlers
           - Ensures unified logging across entire pipeline

        Benefits of dual output:
        - Console: Real-time feedback to user
        - File: Complete record for troubleshooting and reproducibility
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # Generate unique log filename with current timestamp
        # Format: logs/step1_run_20260317_143022.log
        log_file = f'logs/step1_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        # Create the main logger for this runner
        self.logger = logging.getLogger('Step1Runner')
        self.logger.setLevel(logging.INFO)

        # ========== FILE HANDLER ==========
        # Writes detailed logs to file for record-keeping
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # ========== CONSOLE HANDLER ==========
        # Writes minimal output to console for user feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '  %(message)s'  # Just the message, no timestamp/level for clean output
        )
        console_handler.setFormatter(console_formatter)

        # Add both handlers to main logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Configure loggers for all 5 substeps to use same handlers
        # This ensures unified logging across the pipeline
        for name in ['Step1A', 'Step1B', 'Step1C', 'Step1D', 'Step1E']:
            sublogger = logging.getLogger(name)
            sublogger.setLevel(logging.INFO)
            sublogger.addHandler(file_handler)
            sublogger.addHandler(console_handler)

        self.logger.info(f"Logging initialized: {log_file}")

    def check_dependencies(self):
        """
        Verify that all required Python packages are installed.

        This is critical because the pipeline depends on external packages for:
        - Data manipulation (pandas, numpy)
        - Visualization (matplotlib, seaborn)
        - Scientific computing (scipy, sklearn)
        - Single-cell analysis (scanpy, anndata, umap)

        If any package is missing, user gets helpful installation instructions.

        Returns:
            bool: True if all dependencies are satisfied, False otherwise.
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("Checking dependencies...")
        self.logger.info("="*70)

        # List of packages required by the entire pipeline
        required_packages = [
            'pandas',      # Data manipulation (CSV reading, merging, filtering)
            'numpy',       # Numerical arrays and linear algebra
            'matplotlib',  # Low-level plotting (figures)
            'seaborn',     # Statistical data visualization (publication-quality plots)
            'scipy',       # Scientific computing (stats, optimization, clustering)
            'sklearn',     # Machine learning (NMF, metrics, preprocessing)
            'scanpy',      # Single-cell RNA-seq analysis (normalization, UMAP)
            'anndata',     # Annotated data format (standard in single-cell analysis)
            'umap'         # UMAP dimensionality reduction algorithm
        ]

        missing = []

        # Try importing each package
        for package_name in required_packages:
            try:
                __import__(package_name)  # Actually import to verify it's installed
                self.logger.info(f"  [OK] {package_name}")
            except ImportError:
                self.logger.warning(f"  [MISSING] {package_name}")
                missing.append(package_name)

        # If any packages are missing, inform user and exit
        if missing:
            self.logger.error("\nMissing packages detected!")
            self.logger.error("Install with:")
            self.logger.error(f"  pip install {' '.join(missing)}")
            return False

        self.logger.info("All dependencies satisfied.\n")
        return True

    def check_data_availability(self):
        """
        Verify input data is available or will be created.

        This method checks if raw data exists. If not, the individual step modules
        will generate synthetic data automatically. This enables:
        - Testing without real ROSMAP data
        - Running in test mode for rapid validation (21 seconds)
        - Smooth transition when real data becomes available (no code changes needed)

        Returns:
            bool: Always True (synthetic data generation is a fallback)
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("Checking data availability...")
        self.logger.info("="*70)

        raw_data_dir = f"{self.data_dir}/raw"

        # Check if raw data directory exists
        if not os.path.exists(raw_data_dir):
            # If not, synthetic data will be generated by step modules
            self.logger.info(f"  Raw data directory will be created: {raw_data_dir}")
        else:
            # If directory exists, count files and report
            files = os.listdir(raw_data_dir)
            self.logger.info(f"  Found {len(files)} files in {raw_data_dir}")

        self.logger.info("Data availability check complete.\n")
        return True

    def run_step(self, step_name, step_num, step_func):
        """
        Execute a single step with error handling and performance tracking.

        This method:
        1. Calls the step's main() function with appropriate arguments
        2. Times the execution
        3. Catches any exceptions and logs full error context
        4. Records the result (pass/fail) and timing
        5. Returns success status to caller

        Args:
            step_name (str): Descriptive name for logging (e.g., "1A: Data Loading & Preprocessing")
            step_num (int): Step number 1-5 for argument handling
            step_func: The step's main() function to execute

        Returns:
            bool: True if step executed successfully, False if it raised an exception.
                  Note: A "success" means no exception was raised, not necessarily correct results.
        """
        self.logger.info("\n" + "="*70)
        self.logger.info(f"STEP {step_num}/5: {step_name}")
        self.logger.info("="*70)

        step_start = time.time()  # Record start time for performance measurement

        try:
            # Call step function with appropriate arguments based on step number
            # Each step has different requirements (some need skip_deconvolution, some n_subtypes)
            if step_num == 1:
                # Step 1A: Just needs basic arguments
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    test_mode=self.test_mode
                )
            elif step_num == 2:
                # Step 1B: Deconvolution step, needs skip_deconvolution flag
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    skip_deconvolution=self.skip_deconvolution,
                    test_mode=self.test_mode
                )
            elif step_num == 5:
                # Step 1E: NMF clustering, needs n_subtypes override
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    n_subtypes=self.n_subtypes,
                    test_mode=self.test_mode
                )
            else:
                # Steps 1C, 1D, and 1F: Use basic arguments
                result = step_func(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    test_mode=self.test_mode
                )

            # Calculate elapsed time since step started
            step_time = time.time() - step_start

            # Log success and store result
            self.logger.info(f"\n[PASS] {step_name} completed in {step_time:.1f}s")
            self.results[f'step{step_num}'] = {
                'status': 'PASS',
                'time': step_time,
                'result': result
            }

            return True

        except Exception as e:
            # If an exception occurs, log it fully for debugging
            step_time = time.time() - step_start
            self.logger.error(f"\n[FAIL] {step_name} failed after {step_time:.1f}s")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(traceback.format_exc())  # Full stack trace for debugging

            # Store failure result
            self.results[f'step{step_num}'] = {
                'status': 'FAIL',
                'time': step_time,
                'error': str(e)
            }

            return False

    def run_all(self):
        """
        Execute all 5 steps in sequential order.

        Execution flow:
        1. Check dependencies (return False if any missing)
        2. Check data availability (always passes; synthetic data is fallback)
        3. Execute steps 1-5 in order
        4. Stop immediately if any step fails (no point continuing if data is corrupt)
        5. Generate final summary report

        Returns:
            bool: True if all 5 steps succeeded, False if any step failed.
        """
        self.start_time = time.time()  # Record pipeline start time

        self.logger.info("\n\n")
        self.logger.info("#"*70)
        self.logger.info("# Step 1: Patient Stratification & Subtype Discovery")
        self.logger.info("#"*70)

        # Log execution mode information
        if self.test_mode:
            self.logger.info("TEST MODE: Subsampling to 50 patients, 200 proteins")
        else:
            self.logger.info("NORMAL MODE: Using full dataset")

        if self.skip_deconvolution:
            self.logger.info("DECONVOLUTION: Skipping (using equal cell-type proportions)")

        # ========== VERIFY PREREQUISITES ==========
        # Check dependencies first (no point running if packages are missing)
        if not self.check_dependencies():
            self.logger.error("Dependency check failed. Exiting.")
            return False

        # Check data availability (synthetic data fallback, always passes)
        if not self.check_data_availability():
            self.logger.error("Data check failed. Exiting.")
            return False

        # ========== RUN STEPS IN SEQUENCE ==========
        # Define all steps with their names, numbers, and main() functions
        steps = [
            ("1A: Data Loading & Preprocessing", 1, step_1a_main),
            ("1B: Cell-Type Deconvolution", 2, step_1b_main),
            ("1C: Disease Pseudotime", 3, step_1c_main),
            ("1D: AsymAD Resilience Def", 4, step_1d_main),
            ("1E: NMF Consensus Clustering", 5, step_1e_main),
            ("1F: Subtype Validation", 6, step_1f_main),
        ]

        success_count = 0
        for step_name, step_num, step_func in steps:
            if self.run_step(step_name, step_num, step_func):
                success_count += 1
            else:
                # On first failure, stop execution to avoid cascading errors
                # (e.g., data corruption in step 1A will break all downstream steps)
                self.logger.error(f"\nPipeline stopped due to failure in {step_name}")
                return False

        # ========== GENERATE SUMMARY REPORT ==========
        total_time = time.time() - self.start_time
        self._generate_final_report(success_count, total_time)

        return success_count == 6  # True only if all 6 steps passed

    def _generate_final_report(self, success_count, total_time):
        """
        Generate and display final summary report of pipeline execution.

        This report includes:
        - Overall status (SUCCESS or PARTIAL with count)
        - Total runtime in minutes and seconds
        - Per-step results (status, time, key metrics)
        - Output file locations (data, figures, logs)

        This is the last output users see, so it's formatted clearly.

        Args:
            success_count (int): Number of steps that passed (0-5)
            total_time (float): Total pipeline runtime in seconds
        """
        self.logger.info("\n\n")
        self.logger.info("="*70)
        self.logger.info("FINAL SUMMARY REPORT")
        self.logger.info("="*70)

        # ========== OVERALL STATUS ==========
        # Report whether all steps passed or partial execution
        status = "SUCCESS" if success_count == 6 else f"PARTIAL ({success_count}/6)"
        self.logger.info(f"\nOverall Status: {status}")
        self.logger.info(f"Total Runtime: {total_time/60:.1f} minutes ({total_time:.0f}s)")

        # ========== PER-STEP RESULTS ==========
        # For each step 1-6, report its status, timing, and key metrics
        self.logger.info("\nStep Results:")
        for i in range(1, 7):
            step_key = f'step{i}'
            if step_key in self.results:
                result = self.results[step_key]
                status_str = result['status']
                time_str = f"{result['time']:.1f}s"
                step_let = chr(64+i)
                self.logger.info(f"  Step 1{step_let}: {status_str:4s} ({time_str})")

                # If step passed, print key metric from its result
                if status_str == 'PASS' and 'result' in result:
                    result_data = result['result']
                    # Step 1A: Report sample and protein counts
                    if i == 1:
                        self.logger.info(f"        -> {result_data.get('n_samples')} samples x "
                                       f"{result_data.get('n_proteins')} proteins")
                    # Step 1C: Report pseudotime range
                    elif i == 3:
                        self.logger.info(f"        -> Pseudotime range: {result_data.get('pseudotime_min'):.3f} - "
                                       f"{result_data.get('pseudotime_max'):.3f}")
                    # Step 1D (AsymAD): 
                    elif i == 4:
                        self.logger.info(f"        -> AsymAD instances found: {result_data.get('asymad_count')}")
                    # Step 1E: Report number of subtypes and their sizes
                    elif i == 5:
                        sizes = result_data.get('subtype_sizes', {})
                        sizes_str = ', '.join([f"{k}={v}" for k, v in sorted(sizes.items())])
                        self.logger.info(f"        -> k={result_data.get('n_subtypes')}: {sizes_str}")

        # ========== OUTPUT LOCATIONS ==========
        # Tell user where to find the generated files
        self.logger.info(f"\nOutputs saved to:")
        self.logger.info(f"  Data: {os.path.abspath(self.data_dir)}/processed/")
        self.logger.info(f"  Figures: {os.path.abspath(self.results_dir)}/step1/")
        self.logger.info(f"  Logs: {os.path.abspath('logs')}/")

        self.logger.info("\n" + "="*70)
        self.logger.info("Documentation: See README.md for running individual steps")
        self.logger.info("="*70 + "\n")


def main():
    """
    Parse command-line arguments and launch the pipeline.

    This function:
    1. Creates argument parser with help text
    2. Parses command-line arguments
    3. Instantiates Step1Runner with parsed arguments
    4. Runs the full pipeline (run_all())
    5. Exits with appropriate status code (0 for success, 1 for failure)

    Command-line arguments:
    --test: Run in fast test mode (50 samples, 200 proteins, ~30 seconds)
    --skip-deconvolution: Skip cell-type deconvolution (Step 1B)
    --data-dir: Custom path to data folder (default: data/)
    --results-dir: Custom path to results folder (default: results/)
    --n-subtypes: Override automatic k selection (e.g., --n-subtypes 2)
    """
    # Create argument parser with examples
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

    # Define --test argument
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (50 patients, 200 proteins for rapid validation)'
    )

    # Define --skip-deconvolution argument
    parser.add_argument(
        '--skip-deconvolution',
        action='store_true',
        help='Skip Step 1B deconvolution (use equal cell-type proportions instead)'
    )

    # Define --data-dir argument
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data/)'
    )

    # Define --results-dir argument
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Path to results directory (default: results/)'
    )

    # Define --n-subtypes argument
    parser.add_argument(
        '--n-subtypes',
        type=int,
        default=None,
        help='Override automatic k selection (e.g., 2, 3, 4)'
    )

    # Parse arguments from command line
    args = parser.parse_args()

    # Create runner instance and execute pipeline
    runner = Step1Runner(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        test_mode=args.test,
        skip_deconvolution=args.skip_deconvolution,
        n_subtypes=args.n_subtypes
    )
    success = runner.run_all()

    # Exit with appropriate status code
    # sys.exit(0) for success, sys.exit(1) for failure
    # This allows shell scripts to detect success/failure via $?
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
