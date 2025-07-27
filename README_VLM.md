# VLM Training for Image Classification

A comprehensive framework for training Vision Language Models (VLMs) on any image classification dataset using Hugging Face models. This framework supports training small VLMs for image classification tasks with custom prompts and reasoning capabilities.

## Key Features

- **Actual VLMs**: Uses real Vision Language Models from Hugging Face (SmolVLM, Phi-3.5-Vision, Qwen2-VL)
- **Any Dataset Support**: Works with any Hugging Face image classification dataset
- **Auto-Detection**: Automatically detects image/label columns and class names
- **Custom Prompts**: Supports custom system instructions and prompt templates
- **Explainable AI**: VLMs provide reasoning for their classifications
- **Easy Setup**: Simple script-based training with comprehensive logging

## Available VLM Models

| Model | Parameters | Memory | Speed | Quality | Context Length | Description |
|-------|------------|--------|-------|---------|----------------|-------------|
| `HuggingFaceTB/SmolVLM-256M-Instruct` | 256M | ~2GB | Fast | Basic | 2048 | Smallest VLM - fastest inference |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | 500M | ~3GB | Medium | Good | 2048 | Good balance of speed and performance |
| `HuggingFaceTB/SmolVLM-Instruct` | 2.2B | ~8GB | Slow | Best | 2048 | Standard VLM - best performance |
| `microsoft/Phi-3.5-vision-instruct` | 4.2B | ~12GB | Slow | Strong | 4096 | Microsoft Phi-3.5 Vision |
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | ~8GB | Medium | Excellent | 32768 | Excellent for complex reasoning |

## Supported Datasets

The framework works with any Hugging Face image classification dataset. Examples:

- **CIFAR-10**: 10 classes of common objects
- **Food-101**: 101 food categories
- **ImageNet-1K**: 1000 object categories
- **Mirali33/mb-domars16k**: Mars terrain classification (15 classes)

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv mars_env
source mars_env/bin/activate  # On Windows: mars_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Setup

```bash
# Run the test script to verify everything works
python test_vlm_setup.py
```

### 3. Quick Training Test

```bash
# Test with small dataset and few epochs
./run_vlm_training.sh --max-samples 10 --epochs 1
```

### 4. Full Training

```bash
# Train on full dataset
./run_vlm_training.sh --epochs 3 --batch-size 2
```

## Training Options

### Basic Training

```bash
# Default training on Mars dataset
./run_vlm_training.sh

# Custom dataset
./run_vlm_training.sh --dataset "cifar10" --task-description "object classification"

# Different model
./run_vlm_training.sh --model "HuggingFaceTB/SmolVLM-256M-Instruct"
```

### Advanced Options

```bash
# Custom parameters
./run_vlm_training.sh \
  --model "microsoft/Phi-3.5-vision-instruct" \
  --dataset "food101" \
  --batch-size 2 \
  --epochs 5 \
  --learning-rate 5e-6 \
  --use-wandb \
  --task-description "food classification"
```

### Custom Prompts

```bash
# With custom system instructions and prompt template
./run_vlm_training.sh \
  --system-instructions "You are an expert food classifier..." \
  --prompt-template "Analyze this food image and classify it..."
```

## Model Architecture

The framework uses integrated Vision Language Models that process both image and text inputs:

```
Input: [Image] + [Text Prompt]
       ↓
   VLM Model (SmolVLM/Phi-3.5/Qwen2-VL)
       ↓
Output: [Generated Text Response]
       ↓
   Parse: Extract classification from text
```

## Prompt Engineering

The framework uses a conversational format for VLM training:

### System Instructions
```
You are an expert AI assistant specialized in image classification.
Your task is to classify images into one of the following categories:
- class_0
- class_1
- ...
```

### User Prompt Template
```
Please classify this image for the image classification task.

Strictly use this format:
Reasoning: [Provide step-by-step reasoning about what you see in the image]
Answer: [Provide only the class name for the dominant category]
```

### Expected Response
```
Reasoning: Looking at this image, I can see [description]. Based on the visual features, this appears to be [class_name].
Answer: [class_name]
```

## Expected Performance

| Model | Dataset Size | Training Time | Expected Accuracy | Memory Usage |
|-------|-------------|---------------|-------------------|--------------|
| SmolVLM-256M | 10K images | ~30 min | 60-70% | 2GB |
| SmolVLM-500M | 10K images | ~45 min | 70-80% | 3GB |
| SmolVLM-2.2B | 10K images | ~2 hours | 80-90% | 8GB |
| Phi-3.5-Vision | 10K images | ~3 hours | 85-95% | 12GB |

*Times are approximate for CPU training. GPU training is 3-5x faster.*

## Results Structure

Training creates the following structure:

```
vlm_results/
├── best_model/           # Best model during training
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── final_model/          # Final model after training
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── training_metadata.json # Training configuration and results
├── results.json          # Final evaluation results
└── training_curves.png   # Training loss and accuracy plots
```

## Inference

### Single Image Prediction

```python
from inference_vlm import predict_single_image, load_model_and_metadata

# Load trained model
model, processor, metadata = load_model_and_metadata('./vlm_results/best_model')

# Predict on single image
prediction, confidence, full_response = predict_single_image(
    model, processor, 'path/to/image.jpg', metadata
)
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence}")
print(f"Full Response: {full_response}")
```

### Batch Prediction

```python
from inference_vlm import predict_batch

# Predict on multiple images
results = predict_batch(model, processor, image_paths, metadata)
for path, result in results.items():
    print(f"{path}: {result['prediction']} ({result['confidence']:.2f})")
```

## Advanced Usage

### Custom Dataset Format

If your dataset has different column names:

```bash
./run_vlm_training.sh \
  --dataset "your-dataset" \
  --image-column "img" \
  --label-column "category" \
  --class-names "cat" "dog" "bird"
```

### Custom Class Descriptions

Modify the training script to add detailed class descriptions:

```python
class_descriptions = {
    "cat": "a feline animal with whiskers and pointed ears",
    "dog": "a canine animal with four legs and a tail",
    "bird": "a feathered animal with wings and beak"
}
```

### Custom Evaluation

The framework supports custom evaluation metrics by modifying the `evaluate_vlm_model` function in `train_vlm.py`.

## Monitoring Training

### Weights & Biases Integration

```bash
# Enable W&B logging
./run_vlm_training.sh --use-wandb
```

### Local Logging

Training logs include:
- Model loading progress
- Dataset information
- Training loss per step
- Validation accuracy
- Final test results

## Tips for Better Results

1. **Model Selection**: Start with SmolVLM-500M for good balance
2. **Batch Size**: Use smaller batches (2-4) for larger models
3. **Learning Rate**: Start with 1e-5, adjust based on loss curves
4. **Prompt Engineering**: Custom prompts can significantly improve performance
5. **Data Quality**: Ensure images are properly formatted (RGB, reasonable size)
6. **Class Balance**: Consider class imbalance in your dataset

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```bash
# Reduce batch size
./run_vlm_training.sh --batch-size 1

# Use smaller model
./run_vlm_training.sh --model "HuggingFaceTB/SmolVLM-256M-Instruct"
```

**Slow Training**
```bash
# Enable GPU if available
nvidia-smi  # Check GPU availability

# Reduce dataset size for testing
./run_vlm_training.sh --max-samples 100
```

**Model Loading Errors**
```bash
# Check internet connection for model download
# Verify model name is correct
./run_vlm_training.sh --model "HuggingFaceTB/SmolVLM-500M-Instruct"
```

**Dataset Loading Issues**
```bash
# Test dataset loading
python -c "from datasets import load_dataset; ds = load_dataset('Mirali33/mb-domars16k'); print(ds)"
```

### Debug Mode

```bash
# Run with verbose logging
python train_vlm.py --model "HuggingFaceTB/SmolVLM-256M-Instruct" --max-samples 5 --epochs 1
```

## Comparison with Traditional Approaches

| Aspect | Traditional CNN | VLM Approach |
|--------|----------------|--------------|
| **Architecture** | CNN + Classification Head | End-to-end VLM |
| **Input** | Image only | Image + Text |
| **Output** | Class probabilities | Generated text |
| **Explainability** | Limited | Built-in reasoning |
| **Flexibility** | Task-specific | Multi-task capable |
| **Training** | Supervised | Language modeling |
| **Inference** | Direct classification | Text generation + parsing |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.0+
- Accelerate
- Pillow
- scikit-learn
- matplotlib
- wandb (optional)

## Installation

```bash
pip install torch transformers datasets accelerate pillow scikit-learn matplotlib wandb
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test script output
3. Open an issue with detailed error information
4. Include your system configuration and dataset details 