#!/usr/bin/env python3
"""
Vision Language Model (VLM) training for image classification tasks.
Trains small VLMs from Hugging Face on any image classification dataset with custom prompts.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Union
import re

# Set environment variable to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VLM Model configurations - using actual Vision Language Models
VLM_CONFIGS = {
    'HuggingFaceTB/SmolVLM-256M-Instruct': {
        'max_length': 2048,
        'description': 'Smallest VLM (256M params) - fastest inference, basic performance',
        'memory_requirement': 'low'
    },
    'HuggingFaceTB/SmolVLM-500M-Instruct': {
        'max_length': 2048,
        'description': 'Small VLM (500M params) - good balance of speed and performance',
        'memory_requirement': 'medium'
    },
    'HuggingFaceTB/SmolVLM-Instruct': {
        'max_length': 2048,
        'description': 'Standard VLM (2.2B params) - best performance, slower inference',
        'memory_requirement': 'high'
    },
    'microsoft/Phi-3.5-vision-instruct': {
        'max_length': 4096,
        'description': 'Microsoft Phi-3.5 Vision (4.2B params) - strong performance',
        'memory_requirement': 'high'
    },
    'Qwen/Qwen2-VL-2B-Instruct': {
        'max_length': 32768,
        'description': 'Qwen2-VL (2B params) - excellent for complex reasoning',
        'memory_requirement': 'high'
    }
}

def vlm_collate_fn(batch, processor=None):
    """Custom collate function to handle variable-length sequences."""
    # Separate different types of data
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Validate that all items have the required keys
    if not all(item in batch[0] for item in ['input_ids', 'attention_mask', 'pixel_values', 'labels']):
        raise ValueError("Batch items must contain 'input_ids', 'attention_mask', 'pixel_values', and 'labels'")
    
    # Check for empty batch
    if len(batch) == 0:
        raise ValueError("Empty batch provided to collate function")
    
    # Pad input_ids and attention_masks to the same length
    max_length = max(len(ids) for ids in input_ids)
    
    # Debug: print tensor sizes
    if len(input_ids) > 0:
        logger.debug(f"Batch sizes: {[len(ids) for ids in input_ids]}")
        logger.debug(f"Label sizes: {[len(label) for label in labels]}")
        logger.debug(f"Max length: {max_length}")
    
    padded_input_ids = []
    padded_attention_masks = []
    
    # Get padding token from processor or use default
    if processor and hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'pad_token_id'):
        pad_token_id = processor.tokenizer.pad_token_id
    else:
        pad_token_id = 0  # Fallback default
    
    for ids, mask in zip(input_ids, attention_masks):
        # Ensure we have the right data types
        ids = ids.to(torch.long)
        mask = mask.to(torch.long)
        
        # Pad to max_length
        padding_length = max_length - len(ids)
        if padding_length > 0:
            padded_ids = torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
            padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
        else:
            padded_ids = ids
            padded_mask = mask
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    # Pad labels to the same length as input_ids
    padded_labels = []
    for i, label in enumerate(labels):
        try:
            # Ensure we have the right data type
            label = label.to(torch.long)
            
            # Pad to max_length
            padding_length = max_length - len(label)
            if padding_length > 0:
                # Use -100 for padding in labels (this tells the model to ignore these tokens)
                padded_label = torch.cat([label, torch.full((padding_length,), -100, dtype=torch.long)])
            else:
                padded_label = label
            
            padded_labels.append(padded_label)
        except Exception as e:
            logger.error(f"Error processing label {i}: {e}")
            logger.error(f"Label shape: {label.shape if hasattr(label, 'shape') else 'no shape'}")
            logger.error(f"Label type: {type(label)}")
            raise
    
    # Stack all tensors
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'pixel_values': torch.stack(pixel_values),
        'labels': torch.stack(padded_labels)
    }

class VLMDataset(torch.utils.data.Dataset):
    """Generic VLM dataset for any image classification task with custom prompts."""
    
    def __init__(self, dataset, processor, system_instructions: str, prompt_template: str,
                 class_names: List[str], class_descriptions: Optional[Dict[str, str]] = None,
                 image_column: str = 'image', label_column: str = 'label', max_length: int = 2048):
        self.dataset = dataset
        self.processor = processor
        self.system_instructions = system_instructions
        self.prompt_template = prompt_template
        self.class_names = class_names
        self.class_descriptions = class_descriptions or {}
        self.image_column = image_column
        self.label_column = label_column
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        label = item[self.label_column]
        
        # Ensure RGB format
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get class name and description
        class_name = self.class_names[label]
        class_description = self.class_descriptions.get(
            class_name, 
            f"an image of class {class_name}"
        )
        
        # SIMPLIFIED CONVERSATION FORMAT FOR CLASSIFICATION
        # For classification tasks, we use a concise format that focuses on the answer
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt_template}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": class_name}  # Only the class name - most efficient
                ]
            }
        ]
        
        # Apply chat template and process
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        
        # CORRECT VLM FINE-TUNING APPROACH FOR CLASSIFICATION
        # Based on Hugging Face best practices and SmolVLM documentation
        input_ids = inputs['input_ids'].squeeze(0)
        
        # For classification tasks, we should only predict the class name token(s)
        # This is the most stable and efficient approach for VLMs
        
        # Find the assistant response tokens using the chat template structure
        # The processor.apply_chat_template creates a specific structure
        decoded_text = self.processor.decode(input_ids, skip_special_tokens=False)
        
        # Look for the actual class name in the decoded text
        # This ensures we only predict the essential classification tokens
        class_name_start = decoded_text.rfind(class_name)  # Find last occurrence
        
        if class_name_start != -1:
            # Approximate the token position of the class name
            # This is more precise than guessing token counts
            text_before_class = decoded_text[:class_name_start]
            # Rough estimation: average ~4 characters per token
            approx_tokens_before = len(text_before_class) // 4
            
            # Find the actual class name tokens
            assistant_start = max(0, min(approx_tokens_before, len(input_ids) - 10))
            
            # Fine-tune the position by looking for the class name tokens
            class_tokens = self.processor.tokenizer.encode(class_name, add_special_tokens=False)
            
            # Search for class tokens in the last part of the sequence
            search_start = max(0, len(input_ids) - 20)
            for i in range(search_start, len(input_ids) - len(class_tokens) + 1):
                if input_ids[i:i+len(class_tokens)].tolist() == class_tokens:
                    assistant_start = i
                    break
        else:
            # Fallback: predict only the last few tokens (most conservative)
            assistant_start = max(0, len(input_ids) - 5)
        
        logger.debug(f"Sample {idx}: Total tokens={len(input_ids)}, Predicting from token {assistant_start}")
        
        # Create labels - only predict the class name tokens
        labels = input_ids.clone()
        
        # Mask everything except the class name tokens
        labels[:assistant_start] = -100
        
        # Ensure labels have the same length as input_ids
        if len(labels) != len(input_ids):
            logger.warning(f"Label length mismatch for sample {idx}: labels={len(labels)}, input_ids={len(input_ids)}")
            # Truncate or pad labels to match input_ids length
            if len(labels) > len(input_ids):
                labels = labels[:len(input_ids)]
            else:
                # Pad with -100
                padding_length = len(input_ids) - len(labels)
                labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])
        
        # VALIDATION: Ensure we're predicting the right number of tokens for classification
        tokens_to_predict = (labels != -100).sum().item()
        if tokens_to_predict > 10:
            logger.warning(f"Sample {idx}: Predicting {tokens_to_predict} tokens. For classification, fewer is better.")
            # For classification, we should predict very few tokens (just the class name)
            # Force predict only the last few tokens containing the class name
            labels.fill_(-100)
            labels[-5:] = input_ids[-5:]
            tokens_to_predict = 5
        elif tokens_to_predict < 1:
            logger.error(f"Sample {idx}: No tokens to predict! This will cause training issues.")
            # Ensure we predict at least the last token
            labels[-1] = input_ids[-1]
            tokens_to_predict = 1
        
        logger.debug(f"Sample {idx}: Predicting {tokens_to_predict} tokens out of {len(input_ids)} total")
        
        # Debug: check for NaN values
        if torch.isnan(inputs['input_ids']).any():
            logger.warning(f"NaN detected in input_ids for sample {idx}")
        if torch.isnan(inputs['pixel_values']).any():
            logger.warning(f"NaN detected in pixel_values for sample {idx}")
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': labels  # Use the proper labels for VLM training
        }

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

def evaluate_vlm_model(model, dataloader, device, class_names, processor, dataset=None):
    """Evaluate VLM model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Add progress bar for evaluation
    from tqdm import tqdm
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            
            # Generate text responses
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=100,
                do_sample=False
                # Removed temperature=0.0 as it's not needed for deterministic generation
            )
            
            # Decode responses
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Extract predictions from generated text
            batch_predictions = []
            for text in generated_texts:
                predicted_class = extract_class_from_response(text, class_names)
                if predicted_class and predicted_class in class_names:
                    batch_predictions.append(class_names.index(predicted_class))
                else:
                    # Use most common class as fallback if can't extract
                    batch_predictions.append(0)  # Default to first class
            
            all_predictions.extend(batch_predictions)
            
            # Get actual labels from dataset if available
            if dataset is not None:
                # Calculate the starting index for this batch
                batch_size = len(batch_predictions)
                start_idx = batch_idx * batch_size
                
                # Get actual labels from dataset
                batch_labels = []
                for i in range(batch_size):
                    if start_idx + i < len(dataset):
                        item = dataset[start_idx + i]
                        if 'label' in item:
                            batch_labels.append(item['label'])
                        else:
                            batch_labels.append(0)  # Fallback
                    else:
                        batch_labels.append(0)  # Fallback
                
                all_labels.extend(batch_labels)
            else:
                # Fallback: use predictions as ground truth (for testing)
                all_labels.extend(batch_predictions)
    
    # Compute metrics
    # Handle case where we have fewer unique classes than expected
    unique_labels = set(all_labels + all_predictions)
    if len(unique_labels) < len(class_names):
        logger.warning(f"Only {len(unique_labels)} classes found in predictions, expected {len(class_names)}")
        # Use only the classes that appear in the data
        available_classes = sorted(list(unique_labels))
        available_class_names = [class_names[i] for i in available_classes if i < len(class_names)]
    else:
        available_classes = list(range(len(class_names)))
        available_class_names = class_names
    
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=available_class_names,
        labels=available_classes,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'classification_report': report
    }

def plot_training_curves(train_losses, val_accuracies, save_path):
    """Plot training curves."""
    if not train_losses or not val_accuracies:
        logger.warning("No training data to plot")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_default_system_instructions(class_names: List[str], task_description: str = "image classification") -> str:
    """Create default system instructions for any classification task."""
    class_list = "\n".join([f"- {name}" for name in class_names])
    
    return f"""You are an expert AI assistant specialized in {task_description}.
Your task is to classify images into one of the following categories:

{class_list}

You will be provided with an image and asked to classify it.
You must respond with a reasoning explanation followed by your final answer.

Analyze the provided image carefully and provide your classification.
"""

def create_default_prompt_template(task_description: str = "image classification") -> str:
    """Create default prompt template for any classification task."""
    return f"""Please classify this image for the {task_description} task.

Strictly use this format:
Reasoning: [Provide step-by-step reasoning about what you see in the image]
Answer: [Provide only the class name for the dominant category]"""

def auto_detect_dataset_info(dataset, image_column: str = 'image', label_column: str = 'label', 
                           dataset_name: str = None, class_names: List[str] = None):
    """Auto-detect dataset information including class names and descriptions."""
    logger.info("Auto-detecting dataset information...")
    
    # Get a sample to inspect the data
    logger.info("Inspecting dataset sample...")
    sample = dataset[0] if hasattr(dataset, '__getitem__') else next(iter(dataset))
    
    # Auto-detect columns if they exist
    available_columns = list(sample.keys())
    
    # Try common image column names
    image_col_candidates = ['image', 'img', 'picture', 'photo']
    detected_image_col = image_column
    for col in image_col_candidates:
        if col in available_columns:
            detected_image_col = col
            break
    
    # Try common label column names  
    label_col_candidates = ['label', 'labels', 'class', 'category', 'target']
    detected_label_col = label_column
    for col in label_col_candidates:
        if col in available_columns:
            detected_label_col = col
            break
    
    # Get unique labels to determine classes
    all_labels = []
    for item in dataset:
        all_labels.append(item[detected_label_col])
    
    unique_labels = sorted(list(set(all_labels)))
    num_classes = len(unique_labels)
    
    # Use provided class names if available, otherwise generate intelligently
    if class_names is not None:
        if len(class_names) != num_classes:
            logger.warning(f"Provided {len(class_names)} class names but dataset has {num_classes} classes")
            logger.warning("Using provided names and padding with generic names if needed")
            # Pad or truncate to match num_classes
            if len(class_names) < num_classes:
                class_names.extend([f"class_{i}" for i in range(len(class_names), num_classes)])
            else:
                class_names = class_names[:num_classes]
        logger.info("Using provided class names")
    elif all(isinstance(label, int) for label in unique_labels):
        # For integer labels, try to infer meaningful names from dataset metadata
        detected_names = []
        
        # Check if dataset has metadata with class information
        if hasattr(dataset, 'features') and 'label' in dataset.features:
            label_feature = dataset.features['label']
            if hasattr(label_feature, 'names') and label_feature.names:
                detected_names = label_feature.names
                logger.info("Found class names in dataset metadata")
            elif hasattr(label_feature, 'int2str') and callable(label_feature.int2str):
                # Try to get class names from int2str mapping
                try:
                    detected_names = [label_feature.int2str(i) for i in range(num_classes)]
                    logger.info("Generated class names from dataset int2str mapping")
                except Exception:
                    pass
        
        # Fallback to generic names if no meaningful names found
        if not detected_names:
            detected_names = [f"class_{i}" for i in range(num_classes)]
            logger.info("Using generic class names for integer labels")
            logger.info("Tip: Use --class_names to specify actual class names")
        
        class_names = detected_names
    else:
        # String labels - use as-is
        class_names = [str(label) for label in unique_labels]
        logger.info("Using string labels as class names")
    
    logger.info(f"Auto-detected dataset info:")
    logger.info(f"  Image column: {detected_image_col}")
    logger.info(f"  Label column: {detected_label_col}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Class names: {class_names}")
    
    return detected_image_col, detected_label_col, class_names, num_classes

def main():
    parser = argparse.ArgumentParser(description='Train VLM for any image classification task')
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, 
                      choices=list(VLM_CONFIGS.keys()),
                      default='HuggingFaceTB/SmolVLM-500M-Instruct',
                      help='VLM model to use (auto-selects smaller model for low memory)')
    parser.add_argument('--dataset', type=str, default='Mirali33/mb-domars16k',
                      help='HuggingFace dataset name or local path')
    parser.add_argument('--dataset_config', type=str, default=None,
                      help='Dataset configuration name')
    parser.add_argument('--image_column', type=str, default='image',
                      help='Name of image column in dataset')
    parser.add_argument('--label_column', type=str, default='label',
                      help='Name of label column in dataset')
    
    # Class information
    parser.add_argument('--num_classes', type=int, default=None,
                      help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                      help='List of class names (auto-generated if not specified)')
    parser.add_argument('--task_description', type=str, default='image classification',
                      help='Description of the classification task')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (1 recommended for VLM stability)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for VLM fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.5, help='Warmup ratio (extended for VLM stability)')
    
    # System arguments
    parser.add_argument('--output_dir', type=str, default='./vlm_results', 
                      help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use from each split (for testing)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto/cuda/mps/cpu). auto will use GPU if available')
    
    # Custom prompts
    parser.add_argument('--system_instructions', type=str, default=None,
                      help='Custom system instructions')
    parser.add_argument('--prompt_template', type=str, default=None,
                      help='Custom prompt template')
    parser.add_argument('--val_split_ratio', type=float, default=0.2,
                      help='Validation split ratio when no validation set exists')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-select model based on available memory
    try:
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        logger.info(f"System memory: {total_memory:.1f}GB")
        
        # Auto-select smaller model for low memory systems
        if total_memory < 8 and args.model == 'HuggingFaceTB/SmolVLM-500M-Instruct':
            logger.info("Low memory detected - auto-switching to smaller model")
            args.model = 'HuggingFaceTB/SmolVLM-256M-Instruct'
            logger.info(f"Selected model: {args.model}")
        elif total_memory < 16 and args.model in ['HuggingFaceTB/SmolVLM-Instruct', 'microsoft/Phi-3.5-vision-instruct', 'Qwen/Qwen2-VL-2B-Instruct']:
            logger.info("Medium memory detected - auto-switching to medium model")
            args.model = 'HuggingFaceTB/SmolVLM-500M-Instruct'
            logger.info(f"Selected model: {args.model}")
    except ImportError:
        logger.info("psutil not available - using default model selection")
    
    # Handle device selection with memory considerations
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
            logger.info("GPU available, using CUDA")
        elif torch.backends.mps.is_available():
            # Check available memory for MPS
            try:
                import psutil
                total_memory = psutil.virtual_memory().total / (1024**3)  # GB
                if total_memory < 16:  # Less than 16GB RAM
                    logger.warning(f"System has {total_memory:.1f}GB RAM - MPS may run out of memory")
                    logger.info("Using CPU for better stability")
                    args.device = 'cpu'
                else:
                    args.device = 'mps'
                    logger.info("MPS available, using Metal GPU acceleration")
            except ImportError:
                args.device = 'mps'
                logger.info("MPS available, using Metal GPU acceleration")
        else:
            args.device = 'cpu'
            logger.info("No GPU acceleration available, using CPU")
    elif args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            logger.warning("CUDA not available, but MPS is available. Using MPS for GPU acceleration")
            args.device = 'mps'
        else:
            logger.warning("CUDA requested but not available, using CPU")
            args.device = 'cpu'
    
    # Initialize accelerator with device configuration
    if args.device == 'cpu':
        accelerator = Accelerator(cpu=True)
    else:
        accelerator = Accelerator()
    
    device = accelerator.device
    
    # Log accelerator configuration
    logger.info(f"Accelerator device: {device}")
    logger.info(f"Accelerator process index: {accelerator.process_index}")
    logger.info(f"Accelerator is main process: {accelerator.is_main_process}")
    
    # Log device information
    logger.info(f"Device: {device}")
    logger.info(f"Device type: {device.type}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    elif device.type == 'mps':
        logger.info("MPS: Apple Silicon GPU acceleration enabled")
    elif device.type == 'cpu':
        logger.info("CPU: No GPU acceleration available")
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project='vlm-classification', config=vars(args))
    
    # Load dataset with simple error handling
    logger.info(f"Loading dataset: {args.dataset}")
    
    # Quick check if dataset exists (for common datasets)
    if args.dataset.startswith('Mirali33/mb-'):
        logger.info("Detected MarsBench dataset - this may take a moment to download...")
    elif args.dataset.startswith('hf-internal-testing/'):
        logger.info("Using internal test dataset...")
    else:
        logger.info("Loading custom dataset...")
    
    try:
        if args.dataset_config:
            logger.info(f"Loading with config: {args.dataset_config}")
            dataset = load_dataset(args.dataset, args.dataset_config)
        else:
            logger.info("Loading dataset without config...")
            dataset = load_dataset(args.dataset)
        
        logger.info(f"✓ Dataset loaded successfully!")
        logger.info(f"Available splits: {list(dataset.keys())}")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} samples")
            
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Troubleshooting tips:")
        logger.error("1. Check your internet connection")
        logger.error("2. Verify the dataset name is correct")
        logger.error("3. Try with --max_samples 100 to test with a small subset")
        logger.error("4. Check if you need to login: huggingface-cli login")
        
        # Offer to use a test dataset instead
        logger.info("Would you like to try with a small test dataset?")
        logger.info("You can run: python train_vlm.py --dataset cifar10 --max_samples 10")
        return
    
    # Auto-detect dataset information
    train_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    logger.info(f"Using '{train_split}' split for analysis")
    
    image_column, label_column, detected_class_names, num_classes = auto_detect_dataset_info(
        dataset[train_split], args.image_column, args.label_column, args.dataset, args.class_names
    )
    
    logger.info("Dataset analysis completed!")
    
    # Use detected class names or override with user-provided values
    class_names = detected_class_names
    if args.class_names:
        class_names = args.class_names
        num_classes = len(class_names)
    if args.num_classes:
        num_classes = args.num_classes
        if len(class_names) != num_classes:
            class_names = [f"class_{i}" for i in range(num_classes)]
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None:
        logger.info(f"Limiting each split to {args.max_samples} samples for testing")
        for split in dataset.keys():
            if len(dataset[split]) > args.max_samples:
                dataset[split] = dataset[split].select(range(args.max_samples))
                logger.info(f"Limited {split} split to {len(dataset[split])} samples")
    
    # Create system instructions and prompt template
    system_instructions = args.system_instructions or create_default_system_instructions(
        class_names, args.task_description
    )
    prompt_template = args.prompt_template or create_default_prompt_template(
        args.task_description
    )
    
    # Initialize processor and model
    logger.info(f"Loading VLM: {args.model}")
    try:
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(args.model)
        logger.info("✓ Processor loaded successfully!")
        
        # Set appropriate dtype and device_map based on device
        logger.info(f"Loading model for device: {args.device}")
        if args.device == 'cpu':
            logger.info("Loading model with float32 for CPU...")
            model = AutoModelForImageTextToText.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                device_map=None
            )
            logger.info("✓ CPU model loaded successfully!")
        elif args.device == 'mps':
            # Use MPS with float32 (MPS doesn't support float16 well)
            logger.info("Loading model with float32 for MPS...")
            model = AutoModelForImageTextToText.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                device_map=None
            )
            logger.info("✓ MPS model loaded successfully!")
        else:
            # Use CUDA with appropriate dtype - prefer bfloat16 for better numerical stability
            try:
                # Try bfloat16 first (better numerical stability than float16)
                logger.info("Loading model with bfloat16 for CUDA...")
                model = AutoModelForImageTextToText.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
                    device_map=None  # Don't use device_map to avoid accelerate issues
                )
                logger.info("✓ CUDA model loaded with bfloat16 (better numerical stability)")
            except Exception as e:
                logger.warning(f"bfloat16 not supported, falling back to float32: {e}")
                # Fallback to float32 if bfloat16 is not supported
                logger.info("Loading model with float32 fallback...")
                model = AutoModelForImageTextToText.from_pretrained(
                    args.model,
                    torch_dtype=torch.float32,  # Fallback to float32 for stability
                    device_map=None  # Don't use device_map to avoid accelerate issues
                )
                logger.info("✓ CUDA model loaded with float32 fallback")
        
        # Log model configuration
        logger.info(f"Model config: {model.config}")
        logger.info(f"Model has loss function: {hasattr(model, 'compute_loss')}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info(f"Training VLM {args.model} on {args.dataset}")
    logger.info(f"Task: {args.task_description}")
    logger.info(f"Classes ({num_classes}): {class_names}")
    
    # Log training stability recommendations
    logger.info("=" * 60)
    logger.info("PROPER VLM TRAINING SETTINGS:")
    logger.info(f"  Learning Rate: {args.learning_rate} (standard for VLM fine-tuning)")
    logger.info(f"  Warmup Ratio: {args.warmup_ratio} (extended warmup for VLM fine-tuning)")
    logger.info(f"  Batch Size: {args.batch_size} (batch size 1 for maximum stability)")
    logger.info(f"  Gradient Clipping: 0.5 → 1.0 (standard for classification)")
    logger.info(f"  Optimizer: AdamW with eps=1e-5 (prevents division by small numbers)")
    logger.info(f"  Scheduler: Cosine with warmup (better for language model training)")
    logger.info("=" * 60)
    
    # Create datasets
    config = VLM_CONFIGS[args.model]
    
    train_dataset = VLMDataset(
        dataset[train_split], processor, system_instructions, prompt_template,
        class_names, image_column=image_column, label_column=label_column,
        max_length=config['max_length']
    )
    
    # Handle validation split
    val_dataset = None
    test_dataset = None
    
    if 'val' in dataset or 'validation' in dataset:
        val_split = 'val' if 'val' in dataset else 'validation'
        val_dataset = VLMDataset(
            dataset[val_split], processor, system_instructions, prompt_template,
            class_names, image_column=image_column, label_column=label_column,
            max_length=config['max_length']
        )
    elif 'test' in dataset:
        # Use test as validation if no val split
        val_dataset = VLMDataset(
            dataset['test'], processor, system_instructions, prompt_template,
            class_names, image_column=image_column, label_column=label_column,
            max_length=config['max_length']
        )
    else:
        # Split training data if no validation set
        logger.info(f"No validation set found, splitting training data {1-args.val_split_ratio:.0%}/{args.val_split_ratio:.0%}")
        train_val_split = dataset[train_split].train_test_split(test_size=args.val_split_ratio, seed=42)
        train_dataset = VLMDataset(
            train_val_split['train'], processor, system_instructions, prompt_template,
            class_names, image_column=image_column, label_column=label_column,
            max_length=config['max_length']
        )
        val_dataset = VLMDataset(
            train_val_split['test'], processor, system_instructions, prompt_template,
            class_names, image_column=image_column, label_column=label_column,
            max_length=config['max_length']
        )
    
    if 'test' in dataset and ('val' in dataset or 'validation' in dataset):
        test_dataset = VLMDataset(
            dataset['test'], processor, system_instructions, prompt_template,
            class_names, image_column=image_column, label_column=label_column,
            max_length=config['max_length']
        )
    
    # Create dataloaders with appropriate settings for device
    num_workers = 0  # Set to 0 to avoid multiprocessing issues
    pin_memory = args.device not in ['cpu']
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=lambda batch: vlm_collate_fn(batch, processor)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=lambda batch: vlm_collate_fn(batch, processor)
    ) if val_dataset else None
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=lambda batch: vlm_collate_fn(batch, processor)
    ) if test_dataset else None
    
    # Setup training with ultra-conservative settings for VLM fine-tuning
    # Based on Hugging Face best practices for language model fine-tuning
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-5,  # Much larger epsilon for VLM stability (prevents division by small numbers)
        betas=(0.9, 0.999)  # Standard Adam betas (more stable than custom betas)
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Use cosine schedule with warmup for better VLM training stability
    # Based on Hugging Face best practices for language model fine-tuning
    from transformers import get_cosine_schedule_with_warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Prepare with accelerator
    if val_dataloader:
        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )
    else:
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
    
    logger.info(f"Training steps: {total_steps}, Warmup: {warmup_steps}")
    
    # Data sanity check - inspect a sample batch before training
    logger.info("Performing data sanity check...")
    sample_batch = next(iter(train_dataloader))
    
    # Decode and inspect the first sample
    sample_input_ids = sample_batch['input_ids'][0]
    sample_labels = sample_batch['labels'][0]
    
    # Decode the full conversation
    decoded_conversation = processor.decode(sample_input_ids, skip_special_tokens=False)
    logger.info("=" * 60)
    logger.info("SAMPLE CONVERSATION:")
    logger.info("=" * 60)
    logger.info(decoded_conversation)
    logger.info("=" * 60)
    
    # Analyze label masking
    user_tokens = (sample_labels == -100).sum().item()
    assistant_tokens = (sample_labels != -100).sum().item()
    total_tokens = len(sample_labels)
    
    logger.info(f"Label Analysis:")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  User tokens (masked with -100): {user_tokens} ({user_tokens/total_tokens*100:.1f}%)")
    logger.info(f"  Assistant tokens (to predict): {assistant_tokens} ({assistant_tokens/total_tokens*100:.1f}%)")
    
    # Check for potential issues
    if assistant_tokens == 0:
        logger.error("ERROR: No assistant tokens to predict! All labels are -100.")
        logger.error("This will cause training instability. Check label creation logic.")
        return
    elif assistant_tokens < 5:
        logger.warning(f"WARNING: Very few assistant tokens ({assistant_tokens}). This may cause instability.")
    elif user_tokens == 0:
        logger.warning("WARNING: No user tokens masked. The model will try to predict the entire sequence.")
    
    logger.info("Data sanity check completed.")
    logger.info("=" * 60)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    global_step = 0
    
    # Gradient stability tracking
    high_gradient_count = 0
    gradient_history = []
    
    model.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Add progress bar for training
        from tqdm import tqdm
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Forward pass - labels are now properly created in the dataset
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            
            # Debug: log device information for first few steps
            if global_step < 5:
                logger.info(f"Step {global_step} - Input device: {input_ids.device}")
                logger.info(f"Step {global_step} - Model device: {next(model.parameters()).device}")
                logger.info(f"Step {global_step} - Labels shape: {labels.shape}")
                logger.info(f"Step {global_step} - Labels contains -100: {(labels == -100).any()}")
            
            # Check for NaN values in inputs
            if torch.isnan(input_ids).any() or torch.isnan(pixel_values).any():
                logger.warning(f"NaN detected in inputs at step {global_step}")
                continue
            
            # Check device consistency
            model_device = next(model.parameters()).device
            if input_ids.device != model_device:
                logger.warning(f"Device mismatch at step {global_step}: inputs on {input_ids.device}, model on {model_device}")
                continue
                
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            loss = outputs.loss
            
            # Debug: log loss type and value
            if global_step % 100 == 0:
                logger.info(f"Step {global_step} - Loss type: {type(loss)} - Loss value: {loss}")
                logger.info(f"Labels contains -100: {(labels == -100).any()}")
                logger.info(f"Labels valid tokens: {(labels != -100).sum()}")
            
            # Check for problematic loss values
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at step {global_step}")
                logger.error(f"Loss value: {loss}")
                logger.error(f"Input shape: {input_ids.shape}")
                logger.error(f"Labels shape: {labels.shape}")
                logger.error(f"Labels min/max: {labels.min()}/{labels.max()}")
                logger.error(f"Labels contains -100: {(labels == -100).any()}")
                logger.error(f"Labels valid tokens: {(labels != -100).sum()}")
                logger.error("This indicates numerical instability. Consider:")
                logger.error("1. Using float32 instead of bfloat16")
                logger.error("2. Reducing learning rate further")
                logger.error("3. Using smaller batch size")
                continue
            
            # Check for infinite loss
            if torch.isinf(loss):
                logger.error(f"Infinite loss detected at step {global_step}")
                logger.error("This indicates gradient explosion. Training will continue with aggressive clipping.")
                continue
            
            # Check for extremely high loss (potential instability)
            # Based on Hugging Face best practices for language model training
            if loss.item() > 20.0:
                logger.warning(f"High loss detected: {loss.item():.4f} at step {global_step}")
                logger.warning("This may indicate training instability. Consider reducing learning rate.")
            elif loss.item() > 50.0:
                logger.error(f"Very high loss: {loss.item():.4f} at step {global_step}")
                logger.error("Skipping this step to prevent gradient explosion.")
                continue
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping for stability (prevent exploding gradients)
            # Based on Hugging Face best practices for language model training
            if accelerator.sync_gradients:
                # Standard gradient clipping for VLM classification
                # With proper label masking, we can use normal clipping values
                if global_step < 100:
                    max_norm = 0.5    # Conservative in early steps
                elif global_step < 500:
                    max_norm = 1.0    # Standard during training
                else:
                    max_norm = 1.0    # Consistent clipping
                
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                
                # Track gradient stability
                clipping_ratio = grad_norm / max_norm if max_norm > 0 else 0
                gradient_history.append(clipping_ratio)
                
                # Count consecutive high gradient steps (adjusted for VLM)
                if clipping_ratio > 20:  # Reasonable threshold for VLM training
                    high_gradient_count += 1
                else:
                    high_gradient_count = 0
                
                # Enhanced gradient monitoring
                if global_step < 20 or global_step % 50 == 0:
                    logger.info(f"Step {global_step} - Gradient norm: {grad_norm:.4f} (clipped to {max_norm}) - "
                              f"Clipping ratio: {clipping_ratio:.1f}x")
                    
                    # Provide guidance based on gradient behavior
                    if clipping_ratio > 100:
                        logger.error(f"Extreme gradient clipping ({clipping_ratio:.1f}x)! Consider:")
                        logger.error("  • Reducing learning rate")
                        logger.error("  • Increasing warmup steps")
                        logger.error("  • Checking data for anomalies")
                    elif clipping_ratio > 50:
                        logger.warning(f"High gradient clipping ({clipping_ratio:.1f}x). Training may be unstable.")
                    elif clipping_ratio < 1.1 and global_step > 200:
                        logger.info(f"Gradients are well-behaved ({clipping_ratio:.1f}x). Training is stable.")
                
                # Early warning system for persistent instability
                if high_gradient_count >= 10:  # Much lower threshold
                    logger.error(f"Training has been unstable for {high_gradient_count} consecutive steps!")
                    logger.error("This indicates the learning rate is still too high or there are data issues.")
                    logger.error("Consider stopping training and:")
                    logger.error("  • Reducing learning rate by another order of magnitude")
                    logger.error("  • Increasing warmup ratio to 0.6 or higher")
                    logger.error("  • Checking for data preprocessing issues")
                    # Don't automatically stop, but give strong warning
                elif high_gradient_count >= 3 and global_step % 10 == 0:  # Much lower threshold
                    logger.warning(f"Training unstable for {high_gradient_count} steps. Monitor closely.")
                
                # Check for problematic gradients
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.error(f"Invalid gradient norm: {grad_norm} at step {global_step}")
                    continue
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'step': global_step
            })
            
            # Log training more frequently
            if global_step % 10 == 0:
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step} - Train Loss: {loss.item():.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
                if args.use_wandb and accelerator.is_main_process:
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': scheduler.get_last_lr()[0],
                        'global_step': global_step
                    })
            
            # Evaluate
            if global_step % args.eval_steps == 0 and val_dataloader:
                val_metrics = evaluate_vlm_model(model, val_dataloader, device, class_names, processor, val_dataset)
                val_accuracies.append(val_metrics['accuracy'])
                
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step} - Val Acc: {val_metrics['accuracy']:.4f}")
                
                if args.use_wandb and accelerator.is_main_process:
                    wandb.log({
                        'val_accuracy': val_metrics['accuracy'],
                        'val_macro_f1': val_metrics['macro_f1'],
                        'global_step': global_step
                    })
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    if accelerator.is_main_process:
                        model.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                        processor.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                        
                        # Save metadata
                        metadata = {
                            'model_name': args.model,
                            'dataset': args.dataset,
                            'task_description': args.task_description,
                            'num_classes': num_classes,
                            'class_names': class_names,
                            'best_val_accuracy': best_val_accuracy,
                            'system_instructions': system_instructions,
                            'prompt_template': prompt_template,
                            'args': vars(args)
                        }
                        with open(os.path.join(args.output_dir, 'training_metadata.json'), 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        logger.info(f"New best model saved: {best_val_accuracy:.4f}")
                
                model.train()
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    test_metrics = None
    if test_dataloader:
        logger.info("Final evaluation on test set...")
        test_metrics = evaluate_vlm_model(model, test_dataloader, device, class_names, processor, test_dataset)
    elif val_dataloader:
        logger.info("Final evaluation on validation set...")
        test_metrics = evaluate_vlm_model(model, val_dataloader, device, class_names, processor, val_dataset)
    
    if accelerator.is_main_process:
        if test_metrics:
            logger.info(f"Final Results:")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        
        # Save final results
        results = {
            'model': args.model,
            'dataset': args.dataset,
            'task_description': args.task_description,
            'num_classes': num_classes,
            'class_names': class_names,
            'best_val_accuracy': best_val_accuracy,
            'system_instructions': system_instructions,
            'prompt_template': prompt_template,
            'args': vars(args)
        }
        
        if test_metrics:
            results.update({
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1'],
                'test_weighted_f1': test_metrics['weighted_f1'],
            })
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        plot_training_curves(
            train_losses, val_accuracies,
            os.path.join(args.output_dir, 'training_curves.png')
        )
        
        # Save final model
        model.save_pretrained(os.path.join(args.output_dir, 'final_model'))
        processor.save_pretrained(os.path.join(args.output_dir, 'final_model'))
        
        if args.use_wandb and test_metrics:
            wandb.log({
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1'],
                'test_weighted_f1': test_metrics['weighted_f1']
            })
            wandb.finish()
        
        logger.info(f"VLM training completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 