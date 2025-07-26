# Mars Segmentation with CLIP & SigLIP

Extend CLIP and SigLIP image encoders for Mars segmentation tasks. This framework builds upon our classification system to handle semantic segmentation of planetary features like craters, boulders, and other geological formations.

## Supported Segmentation Datasets

### 1. Crater Segmentation
- **[mb-crater_binary_seg](https://huggingface.co/datasets/Mirali33/mb-crater_binary_seg)**: Binary crater segmentation (Background vs Crater)
- **[mb-crater_multi_seg](https://huggingface.co/datasets/Mirali33/mb-crater_multi_seg)**: Multi-class crater segmentation
- **[mb-boulder_seg](https://huggingface.co/datasets/Mirali33/mb-boulder_seg)**: Boulder segmentation

### 2. Custom Datasets
The framework supports any HuggingFace dataset with segmentation format:
- **Binary segmentation**: Background vs foreground features
- **Multi-class segmentation**: Multiple geological features
- **Custom image/mask pairs**: Flexible column naming

## Architecture

### Vision Encoder + Segmentation Decoder
```
Input Image (512x512) 
    ↓
CLIP/SigLIP Vision Encoder 
    ↓
Patch Embeddings (16x16 or 32x32 patches)
    ↓
Segmentation Decoder (ConvTranspose layers)
    ↓
Output Segmentation Mask (512x512)
```

### Key Features
- **Patch-based Processing**: Leverages vision transformer patch embeddings
- **Multi-scale Upsampling**: Progressive upsampling to full resolution
- **Flexible Input Sizes**: Supports 224, 256, 512, or custom sizes
- **Class-agnostic Design**: Works with binary or multi-class segmentation

## Quick Start

### 1. Basic Training
```bash
# Binary crater segmentation
./run_segmentation_training.sh --dataset Mirali33/mb-crater_binary_seg

# Multi-class segmentation
./run_segmentation_training.sh --dataset Mirali33/mb-crater_multi_seg --epochs 20

# Custom target size
./run_segmentation_training.sh --target-size 512 --batch-size 2
```

### 2. Advanced Training
```bash
# Train with SigLIP encoder
python train_segmentation.py \
    --model siglip-base-patch16-224 \
    --dataset Mirali33/mb-boulder_seg \
    --batch_size 4 \
    --num_epochs 15 \
    --target_size 512

# Freeze encoder for faster training
python train_segmentation.py \
    --freeze_encoder \
    --learning_rate 1e-3 \
    --num_epochs 5
```

### 3. Custom Dataset
```bash
# Custom segmentation dataset
python train_segmentation.py \
    --dataset your-username/mars-custom-seg \
    --image_column img \
    --mask_column segmentation_mask \
    --target_size 256
```

### 4. Inference & Visualization
```bash
# Single image with visualization
python inference_segmentation.py \
    --model_path ./segmentation_results/best_model.pth \
    --model_key clip-vit-base-patch32 \
    --image_path mars_crater.jpg \
    --visualize \
    --vis_output crater_segmentation.png

# Batch processing
python inference_segmentation.py \
    --model_path ./segmentation_results/best_model.pth \
    --model_key clip-vit-base-patch32 \
    --image_dir /path/to/mars/images/ \
    --output_file batch_results.json
```

## Available Models

| Model | Patch Size | Hidden Dim | Best For |
|-------|------------|------------|----------|
| `clip-vit-base-patch32` | 32×32 | 768 | Fast training, lower resolution |
| `clip-vit-base-patch16` | 16×16 | 768 | **Recommended** balance |
| `clip-vit-large-patch14` | 14×14 | 1024 | High accuracy, slower |
| `siglip-base-patch16-224` | 16×16 | 768 | Alternative to CLIP |
| `siglip-large-patch16-256` | 16×16 | 1024 | Highest capacity |

## Training Parameters

| Parameter | Description | Default | Segmentation Notes |
|-----------|-------------|---------|-------------------|
| `--model` | Model architecture | `clip-vit-base-patch32` | Choose based on accuracy/speed trade-off |
| `--dataset` | HuggingFace dataset | `Mirali33/mb-crater_binary_seg` | Any segmentation dataset |
| `--target_size` | Input image size | 512 | Higher = better detail, slower training |
| `--batch_size` | Training batch size | 4 | Reduce for larger images/models |
| `--num_epochs` | Number of epochs | 10 | Segmentation needs more epochs |
| `--learning_rate` | Learning rate | 1e-4 | Lower than classification |
| `--freeze_encoder` | Freeze vision encoder | False | True for faster training |

## Performance Metrics

### Evaluation Metrics
- **Pixel Accuracy**: Overall correct pixel classification
- **Mean IoU (mIoU)**: Intersection over Union averaged across classes
- **Class IoU**: IoU for each individual class
- **Loss**: Cross-entropy segmentation loss

### Expected Performance

| Dataset | Task | Accuracy | mIoU | Notes |
|---------|------|----------|------|-------|
| mb-crater_binary_seg | Binary crater | ~98% | ~49% | High accuracy, moderate IoU |
| mb-crater_multi_seg | Multi-class crater | ~85% | ~35% | More challenging |
| mb-boulder_seg | Boulder detection | ~92% | ~42% | Depends on boulder size |

## Visualization Features

### 1. Segmentation Overlay
- Original image with colored mask overlay
- Adjustable transparency (alpha parameter)
- Class-specific color coding

### 2. Probability Heatmaps
- Confidence maps for each class
- Uncertainty visualization
- Threshold analysis

### 3. Class Distribution
- Pixel count statistics
- Percentage breakdown by class
- Visual bar charts

## File Structure

```
├── train_segmentation.py          # Main training script
├── inference_segmentation.py      # Inference and visualization
├── run_segmentation_training.sh   # Training convenience script
├── segmentation_results/          # Training outputs
│   ├── best_model.pth             # Best model weights
│   ├── final_model.pth            # Final epoch weights
│   ├── results.json               # Training metrics
│   └── training_curves.png        # Loss/accuracy plots
└── README_SEGMENTATION.md         # This file
```

## Tips for Better Results

### 1. Data Preprocessing
- **Consistent sizing**: Use same target_size for training and inference
- **Quality masks**: Ensure accurate ground truth segmentation
- **Data augmentation**: Consider rotation, flipping for more robust training

### 2. Training Strategy
- **Start small**: Begin with smaller images (256) then scale up
- **Freeze first**: Try frozen encoder first, then fine-tune if needed
- **Learning rate**: Use lower LR (1e-4) than classification tasks
- **Batch size**: Smaller batches for larger images due to memory

### 3. Model Selection
- **Patch size matters**: Smaller patches (16×16) better for fine details
- **CLIP vs SigLIP**: Both work well, try both for your specific task
- **Size vs speed**: Balance model size with inference speed needs

### 4. Evaluation
- **mIoU is key**: Focus on mean IoU over pixel accuracy
- **Class balance**: Check individual class IoU for imbalanced datasets
- **Visual inspection**: Always visually inspect results on test images

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch_size or target_size
2. **Poor IoU**: Try longer training or different learning rate
3. **Blurry masks**: Use smaller patch size models (patch16 vs patch32)
4. **Class imbalance**: Consider weighted loss functions

### Performance Optimization
- Use `--freeze_encoder` for faster training
- Start with target_size 256, then scale up
- Use mixed precision training for larger models
- Monitor GPU memory usage with batch size

## Integration with Classification

The segmentation framework seamlessly integrates with our classification system:

```bash
# Train classification first
python train.py --dataset Mirali33/mb-domars16k --model clip-vit-base-patch32

# Then train segmentation
python train_segmentation.py --dataset Mirali33/mb-crater_binary_seg --model clip-vit-base-patch32

# Use both for comprehensive Mars analysis
```

This allows for complete Mars surface analysis combining both classification and segmentation capabilities! 