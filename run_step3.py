import sys
import os
import time
import logging
import traceback
from collections import OrderedDict

# Add src directory to path
sys.path.insert(0, os.path.abspath('src'))

from step3.step_3a_feature_engineering import main as data_prep_main
from step3.step_3e_stability import main as stability_main

class Step3Runner:
    def __init__(self, data_dir='data', results_dir='results', test_mode=False):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.test_mode = test_mode
        self.results = OrderedDict()
        self.start_time = 0
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('Step3Runner')

    def run_step(self, step_name, step_num, step_func):
        self.logger.info(f"\n{'='*70}\n[STEP 3{chr(64+step_num)}] {step_name}\n{'='*70}")
        step_start = time.time()
        try:
            result = step_func(self.data_dir, self.results_dir, self.test_mode)
            step_time = time.time() - step_start
            
            self.logger.info(f"\n[PASS] {step_name} completed in {step_time:.1f}s")
            self.results[f'step{step_num}'] = {'status': 'PASS', 'time': step_time, 'result': result}
            return True
        except Exception as e:
            step_time = time.time() - step_start
            self.logger.error(f"\n[FAIL] {step_name} failed after {step_time:.1f}s")
            self.logger.error(f"Error: {e}")
            self.logger.error(traceback.format_exc())
            self.results[f'step{step_num}'] = {'status': 'FAIL', 'time': step_time, 'error': str(e)}
            return False

    def run_all(self):
        self.start_time = time.time()
        self.logger.info("\n\n" + "#"*70)
        self.logger.info("# Step 3: Subtype-Specific GNNs & Stability Testing")
        self.logger.info("#"*70)
        
        steps = [
            ("3A: GNN Feature Engineering", 1, data_prep_main),
            ("3E: GNN Training & Stability Protocol", 5, stability_main) # E is the orchestrator
        ]
        
        success_count = 0
        for step_name, step_num, step_func in steps:
            if self.run_step(step_name, step_num, step_func):
                success_count += 1
            else:
                self.logger.error(f"\nPipeline stopped due to failure in {step_name}")
                return False
                
        self._generate_report(success_count, time.time() - self.start_time)
        return success_count == 2

    def _generate_report(self, success_count, total_time):
        status = "SUCCESS" if success_count == 2 else f"PARTIAL ({success_count}/2)"
        self.logger.info(f"\n\n{'='*70}\nSTEP 3 FINAL SUMMARY REPORT\n{'='*70}")
        self.logger.info(f"\nOverall Status: {status}")
        self.logger.info(f"Total Runtime: {total_time/60:.1f} minutes ({total_time:.0f}s)\nStep Results:")
        
        for i in [1, 5]:
            k = f'step{i}'
            if k in self.results:
                r = self.results[k]
                name = "Feature Engineering" if i == 1 else "Stability Protocol"
                self.logger.info(f"  Step 3{chr(64+i)}: {r['status']:4s} ({r['time']:.1f}s) - {name}")
                if r['status'] == 'PASS' and i == 5:
                    self.logger.info(f"        -> Subtypes analyzed: {r['result'].get('subtypes_analyzed')}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run AD Research Pipeline Step 3')
    parser.add_argument('--test', action='store_true', help='Run in test mode (fast execution)')
    args = parser.parse_args()
    
    runner = Step3Runner(test_mode=args.test)
    success = runner.run_all()
    sys.exit(0 if success else 1)
