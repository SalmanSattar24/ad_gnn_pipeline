"""
Step 1C (Novel Addition 1): Cognitive Resilience Inversion Analysis (AsymAD Definition)

PURPOSE:
This module identifies cognitively resilient individuals (AsymAD) who carry high
AD pathology but remain cognitively normal. This creates a 3-class system
(Healthy Control, AsymAD, Clinical AD) which is critical for downstream resilience
factor discovery.

SCIENTIFIC RATIONALE:
- Most AD target discovery only compares Clinical AD vs Healthy Controls.
- AsymAD patients carry the amyloid/tau burden but resist dementia.
- Identifying proteins that differ between AsymAD and Clinical AD (matched for pathology)
  allows discovery of "resilience factors" (protective targets) rather than just "disease drivers".

CLASSIFICATION LOGIC:
1. Clinical AD: diagnosis == 'AD' (or cogdx >= 2)
2. AsymAD: diagnosis == 'Control' (or cogdx == 1) AND braaksc >= 3 AND ceradsc >= 3
3. Healthy Control: All other cognitively normal individuals (low pathology)

OUTPUTS:
- Updates master_patient_table.csv with a new 'patient_class' column
- Prints group distributions and saves a summary table
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings

warnings.filterwarnings('ignore')

def classify_patients(metadata_df):
    """
    Classify patients into Healthy Control, AsymAD, or Clinical AD.
    """
    # Create the new column with NaN initially
    metadata_df['patient_class'] = np.nan
    
    # Helper to safely extract columns regardless of naming variations
    def get_col(variants):
        for v in variants:
            if v in metadata_df.columns:
                return v
        return None
        
    diag_col = get_col(['diagnosis', 'Diagnosis'])
    cogdx_col = get_col(['cogdx', 'COGDX'])
    braak_col = get_col(['braaksc', 'BRAAKSC', 'braak'])
    cerad_col = get_col(['ceradsc', 'CERADSC', 'cerad'])
    
    if not (diag_col and braak_col and cerad_col):
        logging.warning("Missing necessary clinical columns for AsymAD classification.")
        if diag_col:
            # Fallback: Just use diagnosis if pathology data isn't there
            metadata_df.loc[metadata_df[diag_col] == 'AD', 'patient_class'] = 'Clinical AD'
            metadata_df.loc[metadata_df[diag_col] == 'Control', 'patient_class'] = 'Healthy Control'
        return metadata_df
        
    for idx, row in metadata_df.iterrows():
        diag = row[diag_col]
        braak = row[braak_col]
        cerad = row[cerad_col]
        
        if pd.isna(braak) or pd.isna(cerad):
            if diag == 'AD':
                metadata_df.loc[idx, 'patient_class'] = 'Clinical AD'
            elif diag == 'Control':
                metadata_df.loc[idx, 'patient_class'] = 'Healthy Control'
            continue
            
        cog_normal = (diag == 'Control')
        if cogdx_col and not pd.isna(row[cogdx_col]):
            cog_normal = (row[cogdx_col] == 1) or cog_normal
            
        high_pathology = (braak >= 3) and (cerad >= 3)
        
        if not cog_normal:
            metadata_df.loc[idx, 'patient_class'] = 'Clinical AD'
        else:
            if high_pathology:
                metadata_df.loc[idx, 'patient_class'] = 'AsymAD'
            else:
                metadata_df.loc[idx, 'patient_class'] = 'Healthy Control'
                
    return metadata_df

def save_classifications(master_df, processed_dir):
    master_file = f"{processed_dir}/master_patient_table.csv"
    master_df.to_csv(master_file)
    logging.info(f"  Updated and saved: master_patient_table.csv (added 'patient_class')")
    
    # Save a small summary showing the patient IDs in each class
    class_summary = master_df[['patient_class']].copy()
    class_summary.to_csv(f"{processed_dir}/patient_classes.csv")

def main(data_dir="data", results_dir="results", test_mode=False):
    logger = logging.getLogger("Step1C2_AsymAD")
    
    try:
        processed_dir = f"{data_dir}/processed"
        
        logger.info("="*70)
        logger.info("STEP 1C (Addition): Cognitive Resilience Inversion Analysis")
        logger.info("="*70)
        
        master_file = f"{processed_dir}/master_patient_table.csv"
        
        if not os.path.exists(master_file):
            raise FileNotFoundError(f"Cannot find {master_file}. Run previous steps first.")
            
        master_df = pd.read_csv(master_file, index_col=0)
        logger.info(f"[1/3] Loaded master patient table: {master_df.shape}")
        
        logger.info(f"[2/3] Classifying patients...")
        master_df = classify_patients(master_df)
        
        class_counts = master_df['patient_class'].value_counts()
        logger.info(f"  Class Distribution:")
        for cls_name, count in class_counts.items():
            logger.info(f"    - {cls_name}: {count} patients")
            
        logger.info(f"[3/3] Saving outputs...")
        save_classifications(master_df, processed_dir)
        
        logger.info("="*70)
        logger.info("STEP 1C (Addition) COMPLETE")
        logger.info("="*70)
        
        return {
            'asymad_count': int(class_counts.get('AsymAD', 0)),
            'ad_count': int(class_counts.get('Clinical AD', 0)),
            'control_count': int(class_counts.get('Healthy Control', 0)),
            'status': 'PASS'
        }
        
    except Exception as e:
        logger.error(f"Step 1C (Addition) failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
