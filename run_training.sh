#!/bin/bash

# Clean CLIP/SigLIP Fine-tuning Script for Mars Terrain Classification

echo "CLIP/SigLIP Fine-tuning on Mars Dataset"
echo "======================================="

# Default parameters
MODEL="clip-vit-base-patch32"
BATCH_SIZE=16
NUM_EPOCHS=10
LEARNING_RATE=5e-5
FREEZE_ENCODER=false
USE_WANDB=false
OUTPUT_DIR="./clip_results/$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
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
        --freeze-encoder)
            FREEZE_ENCODER=true
            shift
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL           Model to use (default: clip-vit-base-patch32)"
            echo "  --batch-size SIZE       Batch size (default: 16)"
            echo "  --epochs NUM            Number of epochs (default: 10)"
            echo "  --lr RATE              Learning rate (default: 5e-5)"
            echo "  --freeze-encoder       Freeze vision encoder weights"
            echo "  --use-wandb            Enable Weights & Biases logging"
            echo "  --output-dir DIR       Output directory (default: auto-generated)"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Available models:"
            echo "  clip-vit-base-patch32     - CLIP ViT-B/32 (default)"
            echo "  clip-vit-base-patch16     - CLIP ViT-B/16"
            echo "  clip-vit-large-patch14    - CLIP ViT-L/14"
            echo "  siglip-base-patch16-224   - SigLIP Base"
            echo "  siglip-large-patch16-256  - SigLIP Large"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Freeze Encoder: $FREEZE_ENCODER"
echo "  Use W&B: $USE_WANDB"
echo "  Output: $OUTPUT_DIR"
echo ""

# Build command
CMD="python train.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --output_dir $OUTPUT_DIR"

if [ "$FREEZE_ENCODER" = true ]; then
    CMD="$CMD --freeze_encoder"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
fi

# Run training
eval $CMD

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo ""
    echo "🔍 To run inference:"
    echo "python inference_clip_clean.py \\"
    echo "    --model_path $OUTPUT_DIR/best_model.pth \\"
    echo "    --model_key $MODEL \\"
    echo "    --image_path /path/to/image.jpg \\"
    echo "    --visualize"
    echo ""
    echo "📊 Files created:"
    echo "  - best_model.pth: Best model weights"
    echo "  - final_model.pth: Final model weights"
    echo "  - results.json: Evaluation metrics"
    echo "  - training_curves.png: Training plots"
else
    echo ""
    echo "❌ Training failed! Check the logs above."
    exit 1
fi 