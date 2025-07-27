#!/usr/bin/env python3
"""
Test script to verify VLM training setup works correctly.
"""

import sys
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from PIL import Image
import io
import base64

def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not found")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError:
        print("✗ Datasets not found")
        return False
    
    try:
        import accelerate
        print(f"✓ Accelerate: {accelerate.__version__}")
    except ImportError:
        print("✗ Accelerate not found")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠ CUDA not available, will use CPU")
        return False

def test_model_loading():
    """Test loading a small VLM model."""
    print("\nTesting model loading...")
    
    try:
        model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
        print(f"Loading {model_name}...")
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print("✓ Model loaded successfully")
        return True, model, processor
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False, None, None

def test_dataset_loading():
    """Test loading the Mars dataset."""
    print("\nTesting dataset loading...")
    
    try:
        dataset = load_dataset("Mirali33/mb-domars16k")
        print(f"✓ Dataset loaded: {len(dataset['train'])} train samples")
        
        # Test a sample
        sample = dataset['train'][0]
        print(f"✓ Sample structure: {list(sample.keys())}")
        print(f"✓ Image type: {type(sample['image'])}")
        print(f"✓ Label: {sample['label']}")
        
        return True, dataset
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False, None

def test_inference(model, processor, dataset):
    """Test basic inference."""
    print("\nTesting inference...")
    
    try:
        # Get a sample image
        sample = dataset['train'][0]
        image = sample['image']
        
        # Create a simple prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do you see in this image? Describe it briefly."}
                ]
            }
        ]
        
        # Process
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0
            )
        
        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("✓ Inference successful")
        print(f"Generated text: {generated_text[-100:]}")  # Show last 100 chars
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

def test_training_script():
    """Test if training script can be imported."""
    print("\nTesting training script...")
    
    try:
        import train_vlm
        print("✓ Training script imports successfully")
        return True
    except Exception as e:
        print(f"✗ Training script import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("VLM Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_passed = False
    
    # Test CUDA
    cuda_available = test_cuda()
    
    # Test model loading
    model_success, model, processor = test_model_loading()
    if not model_success:
        all_passed = False
    
    # Test dataset loading
    dataset_success, dataset = test_dataset_loading()
    if not dataset_success:
        all_passed = False
    
    # Test inference (only if model and dataset loaded)
    if model_success and dataset_success:
        if not test_inference(model, processor, dataset):
            all_passed = False
    
    # Test training script
    if not test_training_script():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! VLM setup is ready.")
        print("\nYou can now run:")
        print("  ./run_vlm_training.sh --max-samples 10 --epochs 1")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 