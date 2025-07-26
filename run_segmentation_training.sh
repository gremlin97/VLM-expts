#!/bin/bash

# Mars Segmentation Training Script
# Fine-tune CLIP/SigLIP image encoders for Mars segmentation tasks

set -e

# Default parameters
MODEL="clip-vit-base-patch32"
DATASET="Mirali33/mb-crater_binary_seg"
BATCH_SIZE=4
NUM_EPOCHS=10
LEARNING_RATE=1e-4
TARGET_SIZE=512
OUTPUT_DIR="./segmentation_results"
FREEZE_ENCODER=false
USE_WANDB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --target-size)
            TARGET_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --freeze-encoder)
            FREEZE_ENCODER=true
            shift
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL              Model architecture (default: clip-vit-base-patch32)"
            echo "  --dataset DATASET          HuggingFace dataset name (default: Mirali33/mb-crater_binary_seg)"
            echo "  --batch-size SIZE          Batch size (default: 4)"
            echo "  --epochs EPOCHS            Number of epochs (default: 10)"
            echo "  --lr RATE                  Learning rate (default: 1e-4)"
            echo "  --target-size SIZE         Target image size (default: 512)"
            echo "  --output-dir DIR           Output directory (default: ./segmentation_results)"
            echo "  --freeze-encoder           Freeze the vision encoder"
            echo "  --use-wandb                Enable Weights & Biases logging"
            echo "  --help                     Show this help message"
            echo ""
            echo "Available models:"
            echo "  - clip-vit-base-patch32"
            echo "  - clip-vit-base-patch16"
            echo "  - clip-vit-large-patch14"
            echo "  - siglip-base-patch16-224"
            echo "  - siglip-large-patch16-256"
            echo ""
            echo "Available datasets:"
            echo "  - Mirali33/mb-crater_binary_seg (binary crater segmentation)"
            echo "  - Mirali33/mb-crater_multi_seg (multi-class crater segmentation)"
            echo "  - Mirali33/mb-boulder_seg (boulder segmentation)"
            echo ""
            echo "Examples:"
            echo "  $0 --model clip-vit-base-patch32 --dataset Mirali33/mb-crater_binary_seg"
            echo "  $0 --model siglip-base-patch16-224 --batch-size 8 --epochs 20"
            echo "  $0 --freeze-encoder --epochs 5 --lr 1e-3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Mars Segmentation Training ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Target size: ${TARGET_SIZE}x${TARGET_SIZE}"
echo "Output directory: $OUTPUT_DIR"
echo "Freeze encoder: $FREEZE_ENCODER"
echo "Use W&B: $USE_WANDB"
echo "================================="

# Build command
CMD="python train_segmentation.py \
    --model $MODEL \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --target_size $TARGET_SIZE \
    --output_dir $OUTPUT_DIR"

if [ "$FREEZE_ENCODER" = true ]; then
    CMD="$CMD --freeze_encoder"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
fi

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""

$CMD 