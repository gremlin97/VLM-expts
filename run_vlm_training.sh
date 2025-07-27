#!/bin/bash

# VLM Training Script for Image Classification
# This script trains Vision Language Models on any image classification dataset

set -e

# Default parameters
MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
DATASET="Mirali33/mb-domars16k"
BATCH_SIZE=1
EPOCHS=3
LEARNING_RATE=1e-5
OUTPUT_DIR="./vlm_results"
USE_WANDB=false
MAX_SAMPLES=""
TASK_DESCRIPTION="image classification"
SYSTEM_INSTRUCTIONS=""
PROMPT_TEMPLATE=""
CLASS_NAMES=""

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
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task-description)
            TASK_DESCRIPTION="$2"
            shift 2
            ;;
        --system-instructions)
            SYSTEM_INSTRUCTIONS="$2"
            shift 2
            ;;
        --prompt-template)
            PROMPT_TEMPLATE="$2"
            shift 2
            ;;
        --class-names)
            CLASS_NAMES="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL                   VLM model to use (default: HuggingFaceTB/SmolVLM-500M-Instruct)"
            echo "  --dataset DATASET               HuggingFace dataset name (default: Mirali33/mb-domars16k)"
            echo "  --batch-size SIZE               Batch size (default: 4)"
            echo "  --epochs EPOCHS                 Number of epochs (default: 3)"
            echo "  --lr RATE                       Learning rate (default: 1e-5)"
            echo "  --output-dir DIR                Output directory (default: ./vlm_results)"
            echo "  --task-description DESC         Task description (default: image classification)"
            echo "  --system-instructions TEXT      Custom system instructions"
            echo "  --prompt-template TEXT          Custom prompt template"
            echo "  --class-names NAMES             Custom class names (space-separated)"
            echo "  --use-wandb                     Enable Weights & Biases logging"
            echo "  --max-samples N                 Limit samples for testing"
            echo "  --help                          Show this help message"
            echo ""
            echo "Available VLM models:"
            echo "  - HuggingFaceTB/SmolVLM-256M-Instruct    (256M params - fastest, basic performance)"
            echo "  - HuggingFaceTB/SmolVLM-500M-Instruct    (500M params - good balance, recommended)"
            echo "  - HuggingFaceTB/SmolVLM-Instruct         (2.2B params - best performance, slower)"
            echo "  - microsoft/Phi-3.5-vision-instruct      (4.2B params - strong performance)"
            echo "  - Qwen/Qwen2-VL-2B-Instruct             (2B params - excellent reasoning)"
            echo ""
            echo "Example datasets:"
            echo "  - Mirali33/mb-domars16k                  (Mars terrain classification)"
            echo "  - cifar10                                (General object classification)"
            echo "  - food101                                (Food classification)"
            echo "  - imagenet-1k                            (ImageNet classification)"
            echo ""
            echo "Example with custom prompts:"
            echo "  $0 --dataset Mirali33/mb-domars16k \\"
            echo "      --system-instructions \"You are an expert Martian geologist...\" \\"
            echo "      --prompt-template \"Classify the Martian surface landform...\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python train_vlm.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --num_epochs $EPOCHS"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --task_description \"$TASK_DESCRIPTION\""

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$SYSTEM_INSTRUCTIONS" ]; then
    CMD="$CMD --system_instructions \"$SYSTEM_INSTRUCTIONS\""
fi

if [ -n "$PROMPT_TEMPLATE" ]; then
    CMD="$CMD --prompt_template \"$PROMPT_TEMPLATE\""
fi

if [ -n "$CLASS_NAMES" ]; then
    # Pass class names as a single argument with multiple values
    CMD="$CMD --class_names $CLASS_NAMES"
fi

# Print configuration
echo "=== VLM Training Configuration ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Task: $TASK_DESCRIPTION"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Output Directory: $OUTPUT_DIR"
echo "Use W&B: $USE_WANDB"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples: $MAX_SAMPLES"
fi
if [ -n "$SYSTEM_INSTRUCTIONS" ]; then
    echo "Custom System Instructions: Yes"
fi
if [ -n "$PROMPT_TEMPLATE" ]; then
    echo "Custom Prompt Template: Yes"
fi
if [ -n "$CLASS_NAMES" ]; then
    echo "Class Names: Yes"
fi
echo "================================"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "Warning: CUDA not detected, training will be slower on CPU"
fi

# Check Python and required packages
echo "Checking Python environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo ""
echo "Starting VLM training..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "Training completed! Results saved to $OUTPUT_DIR"
echo ""
echo "To run inference on a single image:"
echo "python inference_vlm.py --model_path $OUTPUT_DIR/best_model --image_path your_image.jpg --visualize"
echo ""
echo "To run batch inference:"
echo "python inference_vlm.py --model_path $OUTPUT_DIR/best_model --image_dir /path/to/images/ --output_file predictions.json" 