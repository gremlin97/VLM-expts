#!/bin/bash

# Mars VLM Training Script with Custom Prompts
# This script trains a VLM on the Mars dataset with specific Martian geologist prompts

set -e

# Activate virtual environment
source mars_env/bin/activate

# Custom system instructions for Mars terrain classification
SYSTEM_INSTRUCTIONS="You are an expert Martian geologist AI. Your task is to classify Martian surface landform images. You will be provided with an image of a Martian surface landform. You must respond with ONLY the three-letter abbreviation of the most prominent landform class present in the image. Only the three-letter abbreviation is required. Here are the possible landform classes, their abbreviations, and their definitions: Aeolian Bedforms: - (ael) Aeolian Curved: Wind-formed bedforms with a curved, dune-like, or rippled appearance. - (aec) Aeolian Straight: Wind-formed bedforms with a straight, linear, or elongated ridge-like appearance. Topographic Landforms: - (cli) Cliff: A steep, near-vertical, or very abrupt rock exposure or slope. - (rid) Ridge: An elongated, narrow elevation or crest of land. - (fsf) Channel: A depression, groove, or trough, often suggesting past fluid flow (e.g., water or lava). - (sfe) Mounds: Distinct, rounded, or irregularly shaped raised landforms or protuberances. Slope Feature Landforms: - (fsg) Gullies: Small, incised channels or ravines, typically found on slopes, potentially formed by fluid or debris flows. - (fse) Slope Streaks: Dark or light markings that appear on slopes, often attributed to dry granular flows or small avalanches. - (fss) Mass Wasting: Features resulting from the downslope movement of rock, regolith, and soil under gravity (e.g., landslides, slumps). Impact Landforms: - (cra) Crater: A bowl-shaped depression, typically circular or sub-circular, formed by an impact event. - (sfx) Crater Field: An area characterized by a significant concentration or cluster of impact craters. Basic Terrain Landforms: - (mix) Mixed Terrain: An area exhibiting a combination of characteristics from multiple distinct landform types, without one single dominant type. - (rou) Rough Terrain: An area characterized by irregular, uneven, broken, or difficult-to-traverse surfaces. - (smo) Smooth Terrain: An area characterized by relatively even, regular surfaces with little to no significant relief or texture. - (tex) Textured Terrain: An area exhibiting a distinct or noticeable surface pattern, fabric, or texture that is not clearly one of the more specific landforms. Analyze the provided image and output only the three-letter abbreviation for the dominant landform."

# Custom prompt template
PROMPT_TEMPLATE="Classify the Martian surface landform in the following image. Strictly use this format: Reasoning: [step-by-step reasoning] Answer: [Provide only the three-letter abbreviation for the dominant landform type]"

# Mars landform class names (mapping from integer labels to actual names)
CLASS_NAMES="ael aec cli rid fsf sfe fsg fse fss cra sfx mix rou smo tex"

# Run training with custom prompts and class names
echo "=== Mars VLM Training with Custom Prompts ==="
echo "Dataset: Mirali33/mb-domars16k"
echo "Task: Mars terrain classification"
echo "Model: HuggingFaceTB/SmolVLM-500M-Instruct"
echo "Custom System Instructions: Yes"
echo "Custom Prompt Template: Yes"
echo "Custom Class Names: Yes"
echo "Class Names: $CLASS_NAMES"
echo "=============================================="

# Build the command
CMD="python train_vlm.py"
CMD="$CMD --model HuggingFaceTB/SmolVLM-500M-Instruct"
CMD="$CMD --dataset Mirali33/mb-domars16k"
CMD="$CMD --batch_size 1"
CMD="$CMD --num_epochs 3"
CMD="$CMD --learning_rate 1e-5"
CMD="$CMD --output_dir ./vlm_results"
CMD="$CMD --task_description \"Mars terrain classification\""
CMD="$CMD --system_instructions \"$SYSTEM_INSTRUCTIONS\""
CMD="$CMD --prompt_template \"$PROMPT_TEMPLATE\""
CMD="$CMD --class_names $CLASS_NAMES"

# Add optional parameters if provided
if [ "$1" = "--test" ]; then
    CMD="$CMD --max_samples 5 --num_epochs 1"
    echo "Running in test mode (5 samples, 1 epoch)"
fi

if [ "$1" = "--quick" ]; then
    CMD="$CMD --max_samples 50 --num_epochs 2"
    echo "Running in quick mode (50 samples, 2 epochs)"
fi

# Print the command
echo "Command: $CMD"
echo ""

# Run training
eval $CMD

echo ""
echo "Mars VLM training completed!"
echo "Results saved to ./vlm_results"
echo ""
echo "To test inference:"
echo "python inference_vlm.py --model_path ./vlm_results/best_model --image_path your_mars_image.jpg" 