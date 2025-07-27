#!/usr/bin/env python3
"""
Comprehensive test script for all Mars datasets with CLIP and SigLIP.
Tests both classification and segmentation tasks with minimal training (1 epoch).
"""

import subprocess
import sys
import json
import os
import time
from typing import Dict, List, Tuple
from datetime import datetime

# Mars datasets for classification
CLASSIFICATION_DATASETS = {
    'Mirali33/mb-atmospheric_dust_cls_edr': 'Binary atmospheric dust classification (EDR)',
    'Mirali33/mb-atmospheric_dust_cls_rdr': 'Binary atmospheric dust classification (RDR)',
    'Mirali33/mb-change_cls_ctx': 'Change detection classification (CTX)',
    'Mirali33/mb-change_cls_hirise': 'Change detection classification (HiRISE)',
    'Mirali33/mb-domars16k': '15-class Mars terrain classification',
    'Mirali33/mb-frost_cls': 'Frost classification',
    'Mirali33/mb-landmark_cls': '8-class Mars landmark classification',
    'Mirali33/mb-surface_cls': 'Surface classification',
    'Mirali33/mb-surface_multi_label_cls': 'Multi-label surface classification'
}

# Mars datasets for segmentation
SEGMENTATION_DATASETS = {
    'Mirali33/mb-boulder_seg': 'Boulder segmentation',
    'Mirali33/mb-conequest_seg': 'Cone/quest segmentation',
    'Mirali33/mb-crater_binary_seg': 'Binary crater segmentation',
    'Mirali33/mb-crater_multi_seg': 'Multi-class crater segmentation',
    'Mirali33/mb-mars_seg_mer': 'Mars segmentation (MER)',
    'Mirali33/mb-mars_seg_msl': 'Mars segmentation (MSL)',
    'Mirali33/mb-mmls': 'Multi-label Mars segmentation',
    'Mirali33/mb-s5mars': 'S5 Mars segmentation'
}

# Models to test
MODELS = {
    'clip-vit-base-patch32': 'CLIP ViT-Base (32x32 patches)',
    'siglip-base-patch16-224': 'SigLIP Base (16x16 patches)'
}

def run_command(cmd: str, description: str, timeout: int = 600) -> Tuple[bool, str, float]:
    """Run a command and return success status, output, and duration."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success: {description} ({duration:.1f}s)")
            return True, result.stdout, duration
        else:
            print(f"❌ Failed: {description} ({duration:.1f}s)")
            print(f"Error: {result.stderr}")
            return False, result.stderr, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ Timeout: {description} ({duration:.1f}s)")
        return False, "Command timed out", duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 Exception: {description} ({duration:.1f}s)")
        print(f"Error: {str(e)}")
        return False, str(e), duration

def test_dataset_loading(dataset_name: str) -> bool:
    """Test if a dataset can be loaded from HuggingFace."""
    test_script = f"""
import sys
from datasets import load_dataset
try:
    dataset = load_dataset('{dataset_name}')
    print(f"✅ Dataset {dataset_name} loaded successfully")
    print(f"   Splits: {{list(dataset.keys())}}")
    if 'train' in dataset:
        print(f"   Train samples: {{len(dataset['train'])}}")
        if len(dataset['train']) > 0:
            sample = dataset['train'][0]
            print(f"   Sample keys: {{list(sample.keys())}}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Failed to load {dataset_name}: {{e}}")
    sys.exit(1)
"""
    
    success, output, duration = run_command(f"python -c \"{test_script}\"", f"Loading dataset {dataset_name}")
    return success

def test_classification_training(dataset_name: str, model: str) -> Tuple[bool, str, float]:
    """Test classification training with minimal settings."""
    output_dir = f"./test_results_classification_{dataset_name.replace('/', '_')}_{model}"
    
    cmd = f"""python train.py \
        --dataset {dataset_name} \
        --model {model} \
        --batch_size 4 \
        --num_epochs 1 \
        --eval_steps 50 \
        --output_dir {output_dir}"""
    
    success, output, duration = run_command(cmd, f"Classification training on {dataset_name} with {model}")
    return success, output, duration

def test_segmentation_training(dataset_name: str, model: str) -> Tuple[bool, str, float]:
    """Test segmentation training with minimal settings."""
    output_dir = f"./test_results_segmentation_{dataset_name.replace('/', '_')}_{model}"
    
    cmd = f"""python train_segmentation.py \
        --dataset {dataset_name} \
        --model {model} \
        --batch_size 2 \
        --num_epochs 1 \
        --eval_steps 50 \
        --target_size 256 \
        --output_dir {output_dir}"""
    
    success, output, duration = run_command(cmd, f"Segmentation training on {dataset_name} with {model}")
    return success, output, duration

def main():
    """Main test function."""
    print("🚀 Comprehensive Mars Dataset Testing with CLIP/SigLIP")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'dataset_loading': {},
        'classification_training': {},
        'segmentation_training': {},
        'summary': {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'total_duration': 0
        }
    }
    
    # Test dataset loading
    print("\n📊 Testing Dataset Loading...")
    all_datasets = {**CLASSIFICATION_DATASETS, **SEGMENTATION_DATASETS}
    
    for dataset_name, description in all_datasets.items():
        print(f"\n--- Testing: {dataset_name} ---")
        print(f"Description: {description}")
        success = test_dataset_loading(dataset_name)
        results['dataset_loading'][dataset_name] = success
        results['summary']['total_tests'] += 1
        if success:
            results['summary']['passed_tests'] += 1
        else:
            results['summary']['failed_tests'] += 1
    
    # Test classification training
    print("\n🎯 Testing Classification Training...")
    for dataset_name, description in CLASSIFICATION_DATASETS.items():
        print(f"\n--- Testing classification: {dataset_name} ---")
        print(f"Description: {description}")
        
        for model in MODELS.keys():
            success, output, duration = test_classification_training(dataset_name, model)
            key = f"{dataset_name}_{model}"
            results['classification_training'][key] = {
                'success': success,
                'output': output,
                'duration': duration
            }
            results['summary']['total_tests'] += 1
            results['summary']['total_duration'] += duration
            if success:
                results['summary']['passed_tests'] += 1
            else:
                results['summary']['failed_tests'] += 1
    
    # Test segmentation training
    print("\n🎨 Testing Segmentation Training...")
    for dataset_name, description in SEGMENTATION_DATASETS.items():
        print(f"\n--- Testing segmentation: {dataset_name} ---")
        print(f"Description: {description}")
        
        for model in MODELS.keys():
            success, output, duration = test_segmentation_training(dataset_name, model)
            key = f"{dataset_name}_{model}"
            results['segmentation_training'][key] = {
                'success': success,
                'output': output,
                'duration': duration
            }
            results['summary']['total_tests'] += 1
            results['summary']['total_duration'] += duration
            if success:
                results['summary']['passed_tests'] += 1
            else:
                results['summary']['failed_tests'] += 1
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    # Dataset loading results
    print("\n📊 Dataset Loading Results:")
    for dataset_name, success in results['dataset_loading'].items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {dataset_name}")
    
    # Classification results
    print("\n🎯 Classification Training Results:")
    for key, result in results['classification_training'].items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = f"({result['duration']:.1f}s)"
        print(f"  {status} {key} {duration}")
    
    # Segmentation results
    print("\n🎨 Segmentation Training Results:")
    for key, result in results['segmentation_training'].items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = f"({result['duration']:.1f}s)"
        print(f"  {status} {key} {duration}")
    
    # Summary statistics
    summary = results['summary']
    success_rate = (summary['passed_tests'] / summary['total_tests']) * 100 if summary['total_tests'] > 0 else 0
    
    print(f"\n📈 Summary Statistics:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Total Duration: {summary['total_duration']/60:.1f} minutes")
    
    # Detailed breakdown
    print(f"\n📊 Detailed Breakdown:")
    print(f"  Classification Datasets: {len(CLASSIFICATION_DATASETS)}")
    print(f"  Segmentation Datasets: {len(SEGMENTATION_DATASETS)}")
    print(f"  Models Tested: {len(MODELS)}")
    print(f"  Classification Tests: {len(CLASSIFICATION_DATASETS) * len(MODELS)}")
    print(f"  Segmentation Tests: {len(SEGMENTATION_DATASETS) * len(MODELS)}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'mars_dataset_test_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    
    # Final status
    print("\n" + "=" * 80)
    if summary['failed_tests'] == 0:
        print("🎉 ALL TESTS PASSED! Mars framework is fully functional!")
    else:
        print(f"⚠️  {summary['failed_tests']} tests failed. Check the output above for details.")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    main() 