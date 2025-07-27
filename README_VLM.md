# VLM Training for Image Classification

A comprehensive framework for training Vision Language Models (VLMs) on any image classification dataset using Hugging Face models. This framework supports training small VLMs for image classification tasks with custom prompts and reasoning capabilities.

## 🚀 **Recent Updates & Fixes**

### ✅ **Fixed Issues**
1. **Class Names Parsing**: Fixed shell script to correctly pass class names as separate arguments
2. **Memory Management**: Added MPS memory management for Apple Silicon GPUs
3. **Custom Collation**: Implemented proper batch collation for variable-length sequences
4. **CPU Training**: Force CPU usage to avoid MPS memory issues
5. **Auto-detection**: Enhanced dataset auto-detection for image/label columns

### 🔧 **Technical Improvements**
- **VLM Architecture**: Proper Vision Language Model implementation using `AutoModelForVision2Seq`
- **Prompt Engineering**: Support for custom system instructions and prompt templates
- **Reasoning Capabilities**: Models provide step-by-step reasoning before classification
- **Batch Processing**: Custom collation function for handling variable-length sequences
- **Memory Optimization**: Environment variables and device management for stable training

## Key Features

- **Actual VLMs**: Uses real Vision Language Models from Hugging Face (SmolVLM, Phi-3.5-Vision, Qwen2-VL)
- **Any Dataset Support**: Works with any Hugging Face image classification dataset
- **Auto-Detection**: Automatically detects image/label columns and class names
- **Custom Prompts**: Supports custom system instructions and prompt templates
- **Explainable AI**: VLMs provide reasoning for their classifications
- **Easy Setup**: Simple script-based training with comprehensive logging
- **Memory Optimized**: CPU training with MPS memory management for Apple Silicon

## Available VLM Models

| Model | Parameters | Memory | Speed | Quality | Context Length | Description |
|-------|------------|--------|-------|---------|----------------|-------------|
| `HuggingFaceTB/SmolVLM-256M-Instruct` | 256M | ~2GB | Fast | Basic | 2048 | Smallest VLM - fastest inference |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | 500M | ~3GB | Medium | Good | 2048 | **Recommended** - good balance |
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
./run_vlm_training.sh --epochs 3 --batch-size 1
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
  --batch-size 1 \
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

### Mars Terrain Classification Example

```bash
# Complete Mars training with expert prompts
./run_vlm_training.sh \
    --dataset "Mirali33/mb-domars16k" \
    --model "HuggingFaceTB/SmolVLM-500M-Instruct" \
    --class-names "ael aec cli rid fsf sfe fsg fse fss cra sfx mix rou smo tex" \
    --system-instructions "You are an expert Martian geologist AI. Your task is to classify Martian surface landform images. You will be provided with an image of a Martian surface landform. You must respond with ONLY the three-letter abbreviation of the most prominent landform class present in the image. Here are the possible landform classes: Aeolian Bedforms: - (ael) Aeolian Curved: Wind-formed bedforms with a curved, dune-like, or rippled appearance. - (aec) Aeolian Straight: Wind-formed bedforms with a straight, linear, or elongated ridge-like appearance. Topographic Landforms: - (cli) Cliff: A steep, near-vertical, or very abrupt rock exposure or slope. - (rid) Ridge: An elongated, narrow elevation or crest of land. - (fsf) Channel: A depression, groove, or trough, often suggesting past fluid flow. - (sfe) Mounds: Distinct, rounded, or irregularly shaped raised landforms. Slope Feature Landforms: - (fsg) Gullies: Small, incised channels or ravines, typically found on slopes. - (fse) Slope Streaks: Dark or light markings that appear on slopes. - (fss) Mass Wasting: Features resulting from the downslope movement of rock. Impact Landforms: - (cra) Crater: A bowl-shaped depression formed by an impact event. - (sfx) Crater Field: An area characterized by a significant concentration of impact craters. Basic Terrain Landforms: - (mix) Mixed Terrain: An area exhibiting a combination of characteristics. - (rou) Rough Terrain: An area characterized by irregular, uneven surfaces. - (smo) Smooth Terrain: An area characterized by relatively even surfaces. - (tex) Textured Terrain: An area exhibiting a distinct surface pattern. Analyze the provided image and output only the three-letter abbreviation for the dominant landform." \
    --prompt-template "Classify the Martian surface landform in the following image. Strictly use this format: Reasoning: [step-by-step reasoning] Answer: [Provide only the three-letter abbreviation for the dominant landform type]" \
    --max-samples 200 \
    --epochs 2 \
    --batch-size 1
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

### Technical Implementation

- **Model Loading**: Uses `AutoProcessor` and `AutoModelForVision2Seq`
- **Batch Processing**: Custom `vlm_collate_fn` for variable-length sequences
- **Memory Management**: CPU training with MPS memory optimization
- **Device Handling**: Automatic CPU fallback for Apple Silicon

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
2. **Batch Size**: Use batch size 1 for memory efficiency
3. **Learning Rate**: Start with 1e-5, adjust based on loss curves
4. **Prompt Engineering**: Custom prompts can significantly improve performance
5. **Data Quality**: Ensure images are properly formatted (RGB, reasonable size)
6. **Class Balance**: Consider class imbalance in your dataset
7. **Memory Management**: Use `--max-samples` for testing on limited hardware

## Troubleshooting

### Common Issues

**MPS Memory Issues (Apple Silicon)**
```bash
# Force CPU training (already implemented)
./run_vlm_training.sh --max-samples 50

# Use smaller model
./run_vlm_training.sh --model "HuggingFaceTB/SmolVLM-256M-Instruct"
```

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

**Class Names Issues**
```bash
# Ensure class names are passed correctly
./run_vlm_training.sh --class-names "class1 class2 class3"
```

### Debug Mode

```bash
# Run with verbose logging
python train_vlm.py --model "HuggingFaceTB/SmolVLM-256M-Instruct" --max-samples 5 --epochs 1
```

### Environment Variables

The framework automatically sets these environment variables for stability:
```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
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
| **Memory Usage** | Low | Higher (CPU training) |
| **Training Speed** | Fast | Slower (CPU training) |

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
- sentencepiece>=0.1.99
- protobuf>=3.20.0

## Installation

```bash
pip install torch transformers datasets accelerate pillow scikit-learn matplotlib wandb sentencepiece protobuf
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

## Recent Changes

### Version 1.1 (Latest)
- ✅ Fixed class names parsing in shell script
- ✅ Added MPS memory management for Apple Silicon
- ✅ Implemented custom collation for variable-length sequences
- ✅ Force CPU training to avoid memory issues
- ✅ Enhanced dataset auto-detection
- ✅ Updated README with comprehensive documentation

### Version 1.0
- 🚀 Initial VLM training framework
- 🚀 Support for multiple VLM models
- 🚀 Custom prompt engineering
- 🚀 Auto-dataset detection
- 🚀 Basic inference capabilities 