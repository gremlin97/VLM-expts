#!/usr/bin/env python3
"""
Inference script for trained VLM on image classification tasks.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional
import re

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoProcessor, AutoModelForVision2Seq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_class_from_response(response: str, class_names: List[str]) -> Optional[str]:
    """Extract the predicted class from VLM response."""
    response_lower = response.lower()
    
    # Look for "Answer: CLASS" pattern first
    answer_pattern = r'answer:\s*([^\n\r.!?]+)'
    match = re.search(answer_pattern, response_lower)
    if match:
        answer_text = match.group(1).strip()
        # Check if any class name is in the answer
        for class_name in class_names:
            if class_name.lower() in answer_text:
                return class_name
    
    # Fallback: look for any class name in the response
    for class_name in class_names:
        if class_name.lower() in response_lower:
            return class_name
    
    return None

def load_model_and_metadata(model_path: str, device: str = 'cuda'):
    """Load trained VLM model and its metadata."""
    
    # Check if it's a directory with saved model or a single file
    if os.path.isdir(model_path):
        # Load from directory
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == 'cuda' else None
        )
        
        # Try to load metadata
        metadata_path = os.path.join(os.path.dirname(model_path), 'training_metadata.json')
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(model_path, 'training_metadata.json')
        
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
    else:
        # Assume it's a Hugging Face model name
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == 'cuda' else None
        )
        metadata = {}
    
    model.eval()
    return model, processor, metadata

def predict_single_image(model, processor, image_path: str, 
                        system_instructions: str, prompt_template: str, 
                        class_names: List[str], device: str = 'cuda'):
    """Predict class for a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Create conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_template}
            ]
        }
    ]
    
    # Apply chat template and process
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    
    if device == 'cuda':
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0
        )
    
    # Decode response
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Extract the assistant's response (after the last "Assistant:" token)
    if "Assistant:" in generated_text:
        assistant_response = generated_text.split("Assistant:")[-1].strip()
    else:
        assistant_response = generated_text.strip()
    
    # Extract predicted class
    predicted_class = extract_class_from_response(assistant_response, class_names)
    
    # Calculate confidence (simplified - just based on text matching)
    confidence = 0.8 if predicted_class else 0.1
    
    return {
        'predicted_class': predicted_class,
        'predicted_class_idx': class_names.index(predicted_class) if predicted_class and predicted_class in class_names else -1,
        'confidence': confidence,
        'full_response': assistant_response,
        'raw_generated_text': generated_text
    }

def predict_batch(model, processor, image_dir: str, 
                 system_instructions: str, prompt_template: str,
                 class_names: List[str], device: str = 'cuda'):
    """Predict classes for a batch of images."""
    results = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for image_path in image_files:
        try:
            result = predict_single_image(
                model, processor, image_path, 
                system_instructions, prompt_template, class_names, device
            )
            result['image_path'] = image_path
            results.append(result)
            
            # Log progress
            logger.info(f"Processed {os.path.basename(image_path)}: {result['predicted_class']} ({result['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results

def visualize_prediction(image_path: str, prediction: Dict, class_names: List[str], 
                        save_path: Optional[str] = None):
    """Visualize prediction with image and results."""
    # Load image
    image = Image.open(image_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show image
    ax1.imshow(image)
    predicted_class = prediction.get('predicted_class', 'Unknown')
    confidence = prediction.get('confidence', 0.0)
    ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.3f}')
    ax1.axis('off')
    
    # Show response text
    full_response = prediction.get('full_response', 'No response available')
    ax2.text(0.05, 0.95, "VLM Response:", transform=ax2.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top')
    ax2.text(0.05, 0.85, full_response, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', wrap=True)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Inference with trained VLM for image classification')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model directory or Hugging Face model name')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, default=None,
                      help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                      help='Directory containing images for batch prediction')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file for batch predictions (JSON)')
    parser.add_argument('--visualize', action='store_true',
                      help='Show visualization for single image prediction')
    
    # Task configuration (can be overridden by metadata)
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                      help='List of class names (loaded from metadata if available)')
    parser.add_argument('--task_description', type=str, default='image classification',
                      help='Description of the classification task')
    parser.add_argument('--system_instructions', type=str, default=None,
                      help='Custom system instructions')
    parser.add_argument('--prompt_template', type=str, default=None,
                      help='Custom prompt template')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model and metadata
    logger.info(f"Loading model from {args.model_path}")
    try:
        model, processor, metadata = load_model_and_metadata(args.model_path, args.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Get configuration from metadata or arguments
    class_names = metadata.get('class_names') or args.class_names
    task_description = metadata.get('task_description') or args.task_description
    system_instructions = metadata.get('system_instructions') or args.system_instructions
    prompt_template = metadata.get('prompt_template') or args.prompt_template
    
    # Create default prompts if not provided
    if not class_names:
        logger.error("No class names provided. Please specify --class_names or ensure model metadata contains class names.")
        return
    
    if not system_instructions:
        class_list = "\n".join([f"- {name}" for name in class_names])
        system_instructions = f"""You are an expert AI assistant specialized in {task_description}.
Your task is to classify images into one of the following categories:

{class_list}

You will be provided with an image and asked to classify it.
You must respond with a reasoning explanation followed by your final answer.

Analyze the provided image carefully and provide your classification."""
    
    if not prompt_template:
        prompt_template = f"""Please classify this image for the {task_description} task.

Strictly use this format:
Reasoning: [Provide step-by-step reasoning about what you see in the image]
Answer: [Provide only the class name for the dominant category]"""
    
    logger.info(f"Task: {task_description}")
    logger.info(f"Classes: {class_names}")
    
    # Single image prediction
    if args.image_path:
        logger.info(f"Predicting for single image: {args.image_path}")
        try:
            prediction = predict_single_image(
                model, processor, args.image_path,
                system_instructions, prompt_template, class_names, args.device
            )
            
            print(f"\nPrediction Results:")
            print(f"  Image: {args.image_path}")
            print(f"  Predicted Class: {prediction['predicted_class']}")
            print(f"  Confidence: {prediction['confidence']:.3f}")
            print(f"  Full Response: {prediction['full_response']}")
            
            if args.visualize:
                visualize_prediction(args.image_path, prediction, class_names)
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
    
    # Batch prediction
    elif args.image_dir:
        logger.info(f"Predicting for images in directory: {args.image_dir}")
        try:
            results = predict_batch(
                model, processor, args.image_dir,
                system_instructions, prompt_template, class_names, args.device
            )
            
            # Print summary results
            print(f"\nBatch Prediction Results:")
            successful_predictions = 0
            for result in results:
                if 'error' in result:
                    print(f"  {os.path.basename(result['image_path'])}: ERROR - {result['error']}")
                else:
                    print(f"  {os.path.basename(result['image_path'])}: {result['predicted_class']} ({result['confidence']:.3f})")
                    successful_predictions += 1
            
            print(f"\nSummary: {successful_predictions}/{len(results)} images processed successfully")
            
            # Save results
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {args.output_file}")
                
        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
    
    else:
        logger.error("Please provide either --image_path or --image_dir")
        return

if __name__ == "__main__":
    main() 