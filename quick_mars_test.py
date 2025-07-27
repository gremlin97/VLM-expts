#!/usr/bin/env python3
"""
Quick test script for Mars datasets - tests a subset of datasets for immediate feedback.
"""

import subprocess
import sys
import time

def run_quick_test():
    """Run quick tests on a subset of datasets."""
    
    # Test a few key datasets
    classification_tests = [
        ('Mirali33/mb-domars16k', 'clip-vit-base-patch32'),
        ('Mirali33/mb-landmark_cls', 'siglip-base-patch16-224'),
        ('Mirali33/mb-atmospheric_dust_cls_edr', 'clip-vit-base-patch32')
    ]
    
    segmentation_tests = [
        ('Mirali33/mb-crater_binary_seg', 'siglip-base-patch16-224'),
        ('Mirali33/mb-boulder_seg', 'clip-vit-base-patch32')
    ]
    
    print("🚀 Quick Mars Dataset Test")
    print("=" * 50)
    
    # Test classification
    print("\n🎯 Testing Classification...")
    for dataset, model in classification_tests:
        print(f"\n--- Testing: {dataset} with {model} ---")
        
        cmd = f"""python train.py \
            --dataset {dataset} \
            --model {model} \
            --batch_size 4 \
            --num_epochs 1 \
            --eval_steps 10 \
            --output_dir ./quick_test_classification_{dataset.replace('/', '_')}_{model}"""
        
        print(f"Running: {cmd}")
        start_time = time.time()
        
        try:
            # Run with real-time output
            result = subprocess.run(cmd, shell=True, timeout=300)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {dataset} with {model} ({duration:.1f}s)")
            else:
                print(f"❌ FAILED: {dataset} with {model} ({duration:.1f}s)")
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {dataset} with {model}")
        except Exception as e:
            print(f"💥 EXCEPTION: {dataset} with {model} - {e}")
    
    # Test segmentation
    print("\n🎨 Testing Segmentation...")
    for dataset, model in segmentation_tests:
        print(f"\n--- Testing: {dataset} with {model} ---")
        
        cmd = f"""python train_segmentation.py \
            --dataset {dataset} \
            --model {model} \
            --batch_size 2 \
            --num_epochs 1 \
            --eval_steps 10 \
            --target_size 256 \
            --output_dir ./quick_test_segmentation_{dataset.replace('/', '_')}_{model}"""
        
        print(f"Running: {cmd}")
        start_time = time.time()
        
        try:
            # Run with real-time output
            result = subprocess.run(cmd, shell=True, timeout=300)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {dataset} with {model} ({duration:.1f}s)")
            else:
                print(f"❌ FAILED: {dataset} with {model} ({duration:.1f}s)")
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {dataset} with {model}")
        except Exception as e:
            print(f"💥 EXCEPTION: {dataset} with {model} - {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Quick test completed!")

if __name__ == "__main__":
    run_quick_test() 