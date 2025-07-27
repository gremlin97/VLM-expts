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
    AutoProcessor, AutoModelForVision2Seq,
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
        'description': 'Smallest VLM (256M params) - fastest inference, basic performance'
    },
    'HuggingFaceTB/SmolVLM-500M-Instruct': {
        'max_length': 2048,
        'description': 'Small VLM (500M params) - good balance of speed and performance'
    },
    'HuggingFaceTB/SmolVLM-Instruct': {
        'max_length': 2048,
        'description': 'Standard VLM (2.2B params) - best performance, slower inference'
    },
    'microsoft/Phi-3.5-vision-instruct': {
        'max_length': 4096,
        'description': 'Microsoft Phi-3.5 Vision (4.2B params) - strong performance'
    },
    'Qwen/Qwen2-VL-2B-Instruct': {
        'max_length': 32768,
        'description': 'Qwen2-VL (2B params) - excellent for complex reasoning'
    }
}

def vlm_collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Separate different types of data
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Debug: print tensor sizes
    if len(input_ids) > 0:
        logger.debug(f"Batch sizes: {[len(ids) for ids in input_ids]}")
    
    # Pad input_ids and attention_masks to the same length
    max_length = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        # Ensure we have the right data types
        ids = ids.to(torch.long)
        mask = mask.to(torch.long)
        
        # Pad to max_length
        padding_length = max_length - len(ids)
        if padding_length > 0:
            padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
            padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
        else:
            padded_ids = ids
            padded_mask = mask
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    # Stack all tensors
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'pixel_values': torch.stack(pixel_values),
        'labels': torch.stack(labels)
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
        
        # Create reasoning based on the class
        reasoning = f"Looking at this image, I can see {class_description}. "
        reasoning += f"Based on the visual features, this appears to be {class_name}."
        
        # Format the full conversation
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
                    {"type": "text", "text": f"Reasoning: {reasoning}\nAnswer: {class_name}"}
                ]
            }
        ]
        
        # Apply chat template and process
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
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

def evaluate_vlm_model(model, dataloader, device, class_names, processor):
    """Evaluate VLM model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate text responses
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0
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
                    # Random guess if can't extract
                    batch_predictions.append(0)
            
            all_predictions.extend(batch_predictions)
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names, 
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

def auto_detect_dataset_info(dataset, image_column: str = 'image', label_column: str = 'label'):
    """Auto-detect dataset information including class names and descriptions."""
    # Get a sample to inspect the data
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
    
    # Generate class names
    if all(isinstance(label, int) for label in unique_labels):
        # Integer labels - generate generic names
        class_names = [f"class_{i}" for i in range(num_classes)]
    else:
        # String labels - use as-is
        class_names = [str(label) for label in unique_labels]
    
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
                      help='VLM model to use')
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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    
    # System arguments
    parser.add_argument('--output_dir', type=str, default='./vlm_results', 
                      help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use from each split (for testing)')
    
    # Custom prompts
    parser.add_argument('--system_instructions', type=str, default=None,
                      help='Custom system instructions')
    parser.add_argument('--prompt_template', type=str, default=None,
                      help='Custom prompt template')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator()
    device = accelerator.device
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project='vlm-classification', config=vars(args))
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        if args.dataset_config:
            dataset = load_dataset(args.dataset, args.dataset_config)
        else:
            dataset = load_dataset(args.dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Please ensure the dataset exists and is accessible")
        return
    
    # Auto-detect dataset information
    train_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    image_column, label_column, class_names, num_classes = auto_detect_dataset_info(
        dataset[train_split], args.image_column, args.label_column
    )
    
    # Override with user-provided values if specified
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
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None  # Force CPU usage
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info(f"Training VLM {args.model} on {args.dataset}")
    logger.info(f"Task: {args.task_description}")
    logger.info(f"Classes ({num_classes}): {class_names}")
    
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
        logger.info("No validation set found, splitting training data 80/20")
        train_val_split = dataset[train_split].train_test_split(test_size=0.2, seed=42)
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
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, pin_memory=False, collate_fn=vlm_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=vlm_collate_fn
    ) if val_dataset else None
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=vlm_collate_fn
    ) if test_dataset else None
    
    # Setup training
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
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
    
    # Training loop
    train_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    global_step = 0
    
    model.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                labels=batch['input_ids']  # For language modeling loss
            )
            loss = outputs.loss
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            num_batches += 1
            global_step += 1
            
            # Log training
            if args.use_wandb and accelerator.is_main_process and global_step % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'global_step': global_step
                })
            
            # Evaluate
            if global_step % args.eval_steps == 0 and val_dataloader:
                val_metrics = evaluate_vlm_model(model, val_dataloader, device, class_names, processor)
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
        test_metrics = evaluate_vlm_model(model, test_dataloader, device, class_names, processor)
    elif val_dataloader:
        logger.info("Final evaluation on validation set...")
        test_metrics = evaluate_vlm_model(model, val_dataloader, device, class_names, processor)
    
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