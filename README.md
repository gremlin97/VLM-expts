# Mars Terrain Classification with CLIP & SigLIP

Fine-tune CLIP and SigLIP image encoders on the [Mars terrain classification dataset](https://huggingface.co/datasets/Mirali33/mb-domars16k) for planetary science research.

## Dataset

The dataset contains 29,718 Mars terrain images across 15 different terrain classes:

- **ael**: Aeolian features
- **rou**: Rough terrain  
- **cli**: Cliffs
- **aec**: Aeolian crater
- **tex**: Textured terrain
- **smo**: Smooth terrain
- **fss**: Frost/snow surfaces
- **rid**: Ridges
- **fse**: Frost/snow edges
- **sfe**: Sand/frost edges
- **fsf**: Frost/snow formations
- **fsg**: Frost/snow gullies
- **sfx**: Sand/frost mixed
- **cra**: Craters
- **mix**: Mixed terrain

## Available Models

| Model | Description | Hidden Size |
|-------|-------------|-------------|
| `clip-vit-base-patch32` | CLIP ViT-B/32 (default) | 768 |
| `clip-vit-base-patch16` | CLIP ViT-B/16 | 768 |
| `clip-vit-large-patch14` | CLIP ViT-L/14 | 1024 |
| `siglip-base-patch16-224` | SigLIP Base | 768 |
| `siglip-large-patch16-256` | SigLIP Large | 1024 |

## Setup

```bash
# Create virtual environment
python -m venv mars_env
source mars_env/bin/activate  # On Windows: mars_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train with Default Settings

```bash
chmod +x run_training.sh
./run_training.sh
```

### 2. Train with Custom Parameters

```bash
# Train CLIP ViT-B/32 (recommended)
./run_training.sh --model clip-vit-base-patch32 --epochs 10 --batch-size 16

# Train SigLIP Base
./run_training.sh --model siglip-base-patch16-224 --epochs 8 --batch-size 12

# Freeze encoder (faster training)
./run_training.sh --freeze-encoder --epochs 5 --lr 1e-3
```

### 3. Run Inference

```bash
# Single image prediction
python inference.py \
    --model_path ./results/best_model.pth \
    --model_key clip-vit-base-patch32 \
    --image_path mars_image.jpg \
    --visualize

# Batch prediction
python inference.py \
    --model_path ./results/best_model.pth \
    --model_key clip-vit-base-patch32 \
    --image_dir /path/to/images/ \
    --output_file predictions.json
```

## Training Options

### Command Line Arguments

```bash
python train.py \
    --model clip-vit-base-patch32 \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 5e-5 \
    --freeze_encoder \
    --use_wandb
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | `clip-vit-base-patch32` |
| `--batch_size` | Training batch size | 32 |
| `--num_epochs` | Number of epochs | 10 |
| `--learning_rate` | Learning rate | 5e-5 |
| `--freeze_encoder` | Freeze vision encoder weights | False |
| `--use_wandb` | Enable W&B logging | False |
| `--output_dir` | Output directory | `./clip_results` |

### Training Strategies

**Full Fine-tuning (Recommended)**
```bash
./run_training.sh --model clip-vit-base-patch32 --epochs 10 --lr 5e-5
```

**Frozen Encoder (Faster)**
```bash
./run_training.sh --freeze-encoder --epochs 5 --lr 1e-3
```

**Large Model**
```bash
./run_training.sh --model clip-vit-large-patch14 --batch-size 8 --epochs 8
```

## Expected Performance

Based on the dataset characteristics:

- **Training samples**: 11,305 images
- **Validation samples**: 3,231 images  
- **Test samples**: 1,614 images
- **Expected accuracy**: 88-94% (depending on model and strategy)
- **Training time**: 30 minutes - 2 hours (depending on model size and strategy)

## Model Architecture

The fine-tuned model consists of:

1. **Vision Encoder**: Pretrained CLIP/SigLIP vision encoder
2. **Classification Head**: Two-layer MLP with dropout

```
Input Image (224x224x3)
       ↓
Vision Encoder (CLIP/SigLIP)
       ↓
Visual Features (768/1024-dim)
       ↓
Classification Head
  - Dropout(0.1)
  - Linear(hidden → hidden/2) + ReLU
  - Dropout(0.1)  
  - Linear(hidden/2 → 15)
       ↓
Terrain Class Logits
```

## Results Structure

After training:

```
results/
├── best_model.pth        # Best model weights
├── final_model.pth       # Final model weights
├── results.json          # Evaluation metrics
└── training_curves.png   # Training plots
```

## Monitoring Training

### Using Weights & Biases

```bash
# Login (first time only)
wandb login

# Train with logging
./run_training.sh --use-wandb
```

### Console Output

Monitor training progress through:
- Training loss (should decrease)
- Validation accuracy (should increase)
- Learning rate schedule

## Tips for Better Results

### Model Selection
- **CLIP ViT-B/32**: Best starting point, good balance
- **SigLIP Base**: Often performs better than CLIP
- **Large models**: Higher accuracy but slower training

### Training Tips
- **Learning Rate**: 5e-5 for full fine-tuning, 1e-3 for frozen encoder
- **Batch Size**: Use largest that fits in memory
- **Epochs**: 8-12 typically sufficient
- **Frozen vs Full**: Frozen is faster but slightly lower accuracy

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
./run_training.sh --batch-size 8
```

**Slow Training**
```bash
# Use frozen encoder
./run_training.sh --freeze-encoder
```

**Dataset Loading Issues**
- Ensure stable internet connection for dataset download
- Dataset will be cached after first download

### Debug Mode

```bash
# Quick test run
python train.py --num_epochs 1 --batch_size 4 --eval_steps 10
```

## File Structure

```
VLM/
├── train.py              # Main training script
├── inference.py          # Inference script
├── run_training.sh       # Training helper script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## License

MIT License. The Mars dataset is licensed under CC-BY-4.0.

## Citation

```bibtex
@misc{mars-clip-siglip,
  title={Mars Terrain Classification with CLIP and SigLIP},
  year={2025},
  url={https://huggingface.co/datasets/Mirali33/mb-domars16k}
}
``` 