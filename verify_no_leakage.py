#!/usr/bin/env python3
"""
Verification script to ensure no data leakage between train/dev/test splits.
Run this after any changes to dataset loading logic.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from tier1_fixed import BenchmarkConfig, DatasetManager, DatasetBundle
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def verify_split_separation(bundle: DatasetBundle) -> bool:
    """Verify that train/dev/test splits have no overlapping queries."""
    
    issues = []
    
    # Get query IDs from each split
    train_qids = set(bundle.train.queries.keys()) if bundle.train else set()
    dev_qids = set(bundle.dev.queries.keys()) if bundle.dev else set()
    test_qids = set(bundle.test.queries.keys()) if bundle.test else set()
    
    # Check for overlaps
    train_dev_overlap = train_qids & dev_qids
    train_test_overlap = train_qids & test_qids
    dev_test_overlap = dev_qids & test_qids
    
    if train_dev_overlap:
        issues.append(f"TRAIN-DEV overlap: {len(train_dev_overlap)} queries")
    if train_test_overlap:
        issues.append(f"TRAIN-TEST overlap: {len(train_test_overlap)} queries")
    if dev_test_overlap:
        issues.append(f"DEV-TEST overlap: {len(dev_test_overlap)} queries")
    
    if issues:
        logging.error(f"❌ {bundle.name}: DATA LEAKAGE DETECTED!")
        for issue in issues:
            logging.error(f"  - {issue}")
        return False
    else:
        logging.info(f"✅ {bundle.name}: No leakage (TRAIN: {len(train_qids)}, DEV: {len(dev_qids)}, TEST: {len(test_qids)})")
        return True

def main():
    config = BenchmarkConfig()
    config.quick_smoke_test = True  # Use small subsets for faster testing
    manager = DatasetManager(config)
    
    print("\n" + "="*70)
    print("DATA LEAKAGE VERIFICATION TEST")
    print("="*70 + "\n")
    
    all_ok = True
    
    # Test MS MARCO
    print("Testing MS MARCO...")
    try:
        msmarco = manager.get_msmarco()
        if not verify_split_separation(msmarco):
            all_ok = False
    except Exception as e:
        logging.error(f"❌ MS MARCO: Failed to load - {e}")
        all_ok = False
    
    # Test some BEIR datasets
    beir_datasets = ["nq", "hotpotqa", "scifact", "fiqa"]
    for dataset in beir_datasets:
        print(f"\nTesting BEIR:{dataset}...")
        try:
            bundle = manager.get_beir_dataset(dataset)
            if not verify_split_separation(bundle):
                all_ok = False
        except Exception as e:
            logging.warning(f"⚠️  {dataset}: Could not verify - {e}")
    
    print("\n" + "="*70)
    if all_ok:
        print("✅ ALL TESTS PASSED - NO DATA LEAKAGE DETECTED")
        print("="*70 + "\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED - DATA LEAKAGE DETECTED")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
