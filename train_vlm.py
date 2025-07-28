#!/usr/bin/env python3
"""
VLM Image Classification - Efficient Fine-tuning Approach
Uses modern VLM optimization techniques for classification tasks.
Implements HuggingFace best practices for VLM fine-tuning.
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
import torch.nn as nn
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
    get_cosine_schedule_with_warmup
)
from accelerate import Accelerator
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VLM Model Configurations
VLM_CONFIGS = {
    'HuggingFaceTB/SmolVLM-256M-Instruct': {
        'max_length': 1200,  # Sufficient for image tokens + efficient text
        'description': 'SmolVLM-256M - Efficient classification model',
        'memory_requirement': 'low',
        'optimization_level': 'high'
    },
    'HuggingFaceTB/SmolVLM-500M-Instruct': {
        'max_length': 1400,  # Optimized length with room for images
        'description': 'SmolVLM-500M - Balanced performance and speed',
        'memory_requirement': 'medium',
        'optimization_level': 'high'
    },
    'microsoft/Phi-3.5-vision-instruct': {
        'max_length': 1600,  # Efficient for Phi-3.5 with image support
        'description': 'Phi-3.5 Vision - Enterprise-grade VLM',
        'memory_requirement': 'high',
        'optimization_level': 'high'
    }
}

class VLMClassifier(nn.Module):
    """
    VLM Classifier with efficient fine-tuning approach.
    Maintains VLM architecture while implementing optimized training.
    """
    
    def __init__(self, base_vlm_model, num_classes: int, class_names: List[str]):
        super().__init__()
        self.base_vlm_model = base_vlm_model
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Freeze most of the VLM model for efficiency - only train the last layers
        self._freeze_base_model()
        
        # Get hidden dimension from VLM model
        if hasattr(base_vlm_model.config, 'text_config'):
            self.hidden_dim = base_vlm_model.config.text_config.hidden_size
        else:
            self.hidden_dim = 768  # Fallback
        
        # Classification head - optimized for VLM features
        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),  # GELU for better VLM compatibility
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
        
        # Initialize classification head
        self._init_classification_head()
    
    def _freeze_base_model(self):
        """Freeze most of the base VLM model for efficiency."""
        # Freeze all parameters first
        for param in self.base_vlm_model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the last few layers for fine-tuning
        if hasattr(self.base_vlm_model, 'language_model'):
            # Unfreeze last 2 transformer layers
            if hasattr(self.base_vlm_model.language_model, 'model') and hasattr(self.base_vlm_model.language_model.model, 'layers'):
                layers = self.base_vlm_model.language_model.model.layers
                for layer in layers[-2:]:  # Last 2 layers
                    for param in layer.parameters():
                        param.requires_grad = True
        
        logger.info("Base VLM model frozen - training only last layers and classification head")
    
    def _init_classification_head(self):
        """Initialize classification head with proper scaling."""
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        """
        Forward pass - VLM features + direct classification.
        Maintains VLM architecture while optimizing for classification.
        """
        # Get VLM features with efficient computation
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            vlm_outputs = self.base_vlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract features from the last hidden state
        last_hidden_state = vlm_outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        # Pooling strategy - focus on the last meaningful tokens
        # Use attention mask to find the last non-padded token for each sequence
        batch_size, seq_len, hidden_dim = last_hidden_state.shape
        
        # Find last non-padded position for each sequence
        last_positions = attention_mask.sum(dim=1) - 1  # [batch_size]
        last_positions = last_positions.clamp(min=0, max=seq_len-1)
        
        # Extract features from last positions
        batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
        pooled_features = last_hidden_state[batch_indices, last_positions]  # [batch_size, hidden_dim]
        
        # Direct classification through head
        logits = self.classification_head(pooled_features)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled_features
        }

class VLMDataset(torch.utils.data.Dataset):
    """
    VLM dataset with efficient processing.
    Optimized prompts and tokenization for maximum speed.
    """
    
    def __init__(self, dataset, processor, class_names: List[str],
                 image_column: str = 'image', label_column: str = 'label', max_length: int = 1200,
                 system_instructions: str = None, prompt_template: str = None):
        self.dataset = dataset
        self.processor = processor
        self.class_names = class_names
        self.image_column = image_column
        self.label_column = label_column
        self.max_length = max_length
        self.system_instructions = system_instructions
        self.prompt_template = prompt_template or "Classify the Martian surface landform in the following image. Strictly use this format: Reasoning: [step-by-step reasoning] Answer: [Provide only the three-letter abbreviation for the dominant landform type]"
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        label = item[self.label_column]
        
        # Ensure RGB format
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get class name
        class_name = self.class_names[label]
        
        # Build conversation with system instructions if provided
        messages = []
        
        # Add system instructions if provided
        if self.system_instructions:
            messages.append({
                "role": "system",
                "content": self.system_instructions
            })
        
        # Add user message with image and prompt
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt_template}
            ]
        })
        
        # Add assistant response with class name
        messages.append({
            "role": "assistant", 
            "content": [
                {"type": "text", "text": class_name}
            ]
        })
        
        # Apply chat template with efficiency
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        
        # Processing - Optimized for speed and memory
        inputs = self.processor(
            text=prompt, 
            images=[image], 
            return_tensors="pt",
            max_length=self.max_length,  # Efficient sequences
            truncation=True,
            padding=False  # Dynamic padding in collate_fn
        )
        
        # Label strategy - Predict only the class tokens
        input_ids = inputs['input_ids'].squeeze(0)
        
        # Find the class name tokens efficiently
        class_tokens = self.processor.tokenizer.encode(class_name, add_special_tokens=False)
        
        # Create labels - predict only the essential class tokens
        labels = torch.full_like(input_ids, -100)  # Mask everything
        
        # Find class tokens in the sequence (search from the end for efficiency)
        if len(class_tokens) > 0:
            seq_len = len(input_ids)
            class_len = len(class_tokens)
            
            # Search for class tokens in the last part of the sequence
            for i in range(max(0, seq_len - 20), seq_len - class_len + 1):
                if i + class_len <= seq_len:
                    if input_ids[i:i+class_len].tolist() == class_tokens:
                        labels[i:i+class_len] = input_ids[i:i+class_len]
                        break
            else:
                # Fallback: predict last token
                labels[-1] = input_ids[-1]
        
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': labels,
            'class_label': torch.tensor(label, dtype=torch.long)  # For evaluation
        }

def vlm_collate_fn(batch, processor=None):
    """
    VLM collate function with XLA-optimized padding.
    Based on HuggingFace best practices for maximum efficiency.
    """
    # Extract components
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    class_labels = [item['class_label'] for item in batch]
    
    # PADDING STRATEGY - XLA optimized
    # Pad to multiple of 32 for XLA efficiency (HuggingFace best practice)
    max_length = max(len(ids) for ids in input_ids)
    padded_length = ((max_length + 31) // 32) * 32  # Round up to multiple of 32
    
    # Get padding token
    pad_token_id = processor.tokenizer.pad_token_id if processor and hasattr(processor, 'tokenizer') else 0
    
    # Efficient padding
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    
    for ids, mask, label in zip(input_ids, attention_masks, labels):
        # Calculate padding needed
        padding_length = padded_length - len(ids)
        
        if padding_length > 0:
            # Pad efficiently
            padded_ids = torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
            padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
            padded_label = torch.cat([label, torch.full((padding_length,), -100, dtype=torch.long)])
        else:
            padded_ids = ids
            padded_mask = mask
            padded_label = label
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        padded_labels.append(padded_label)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'pixel_values': torch.stack(pixel_values),
        'labels': torch.stack(padded_labels),
        'class_labels': torch.stack(class_labels)  # For evaluation
    }

def evaluate_vlm_model(model, dataloader, device, class_names):
    """
    VLM evaluation with direct classification.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    from tqdm import tqdm
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            class_labels = batch['class_labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            logits = outputs['logits']
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(class_labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Handle case where not all classes appear
    unique_labels = set(all_labels + all_predictions)
    if len(unique_labels) < len(class_names):
        available_classes = sorted(list(unique_labels))
        available_class_names = [class_names[i] for i in available_classes if i < len(class_names)]
    else:
        available_classes = list(range(len(class_names)))
        available_class_names = class_names
    
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

def auto_detect_dataset_info(dataset, image_column: str = 'image', label_column: str = 'label', 
                           dataset_name: str = None, class_names: List[str] = None):
    """Auto-detect dataset information including class names and descriptions."""
    logger.info("Auto-detecting dataset information...")
    
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
    
    # Use provided class names if available, otherwise generate intelligently
    if class_names is not None:
        if len(class_names) != num_classes:
            logger.warning(f"Provided {len(class_names)} class names but dataset has {num_classes} classes")
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
                try:
                    detected_names = [label_feature.int2str(i) for i in range(num_classes)]
                    logger.info("Generated class names from dataset int2str mapping")
                except Exception:
                    pass
        
        # Fallback to generic names if no meaningful names found
        if not detected_names:
            detected_names = [f"class_{i}" for i in range(num_classes)]
            logger.info("Using generic class names for integer labels")
        
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
    parser = argparse.ArgumentParser(description='VLM Image Classification - Efficient Fine-tuning Approach')
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, 
                      choices=list(VLM_CONFIGS.keys()),
                      default='HuggingFaceTB/SmolVLM-256M-Instruct',
                      help='VLM model for classification')
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
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    
    # Prompt configuration
    parser.add_argument('--system_instructions', type=str, default=None,
                      help='System instructions for the VLM')
    parser.add_argument('--prompt_template', type=str, default=None,
                      help='Prompt template for classification')
    
    # System arguments
    parser.add_argument('--output_dir', type=str, default='./vlm_results', 
                      help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--eval_steps', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use from each split (for testing)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto/cuda/mps/cpu)')
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
        
        if total_memory < 8:
            logger.info("Low memory detected - using most efficient VLM")
            args.model = 'HuggingFaceTB/SmolVLM-256M-Instruct'
    except ImportError:
        logger.info("psutil not available - using default model selection")
    
    # Handle device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
            logger.info("GPU available, using CUDA")
        elif torch.backends.mps.is_available():
            args.device = 'mps'
            logger.info("MPS available, using Metal GPU acceleration")
        else:
            args.device = 'cpu'
            logger.info("No GPU acceleration available, using CPU")
    
    # Initialize accelerator
    if args.device == 'cpu':
        accelerator = Accelerator(cpu=True)
    else:
        accelerator = Accelerator(mixed_precision='fp16' if args.device == 'cuda' else 'no')
    
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project='vlm-classification', config=vars(args))
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        if args.dataset_config:
            dataset = load_dataset(args.dataset, args.dataset_config)
        else:
            dataset = load_dataset(args.dataset)
        logger.info(f"Dataset loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Auto-detect dataset information
    train_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    image_column, label_column, detected_class_names, num_classes = auto_detect_dataset_info(
        dataset[train_split], args.image_column, args.label_column, args.dataset, args.class_names
    )
    
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
    
    # Load VLM processor and model
    logger.info(f"Loading VLM: {args.model}")
    try:
        processor = AutoProcessor.from_pretrained(args.model)
        base_vlm_model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=torch.float32 if args.device == 'cpu' else torch.bfloat16,
            device_map=None
        )
        logger.info("Base VLM model loaded successfully!")
        
        # Create VLM classifier
        model = VLMClassifier(base_vlm_model, num_classes, class_names)
        logger.info("VLM Classifier created!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info("=" * 60)
    logger.info("VLM CLASSIFICATION APPROACH:")
    logger.info(f"  Architecture: VLM with parameter-efficient fine-tuning")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Classes: {num_classes}")
    logger.info(f"  Max Length: {VLM_CONFIGS[args.model]['max_length']}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Approach: VLM with direct classification head")
    logger.info(f"  Optimizations: Parameter-efficient training with XLA-optimized padding")
    logger.info("=" * 60)
    
    # Create datasets
    config = VLM_CONFIGS[args.model]
    
    train_dataset = VLMDataset(
        dataset[train_split], processor, class_names,
        image_column=image_column, label_column=label_column,
        max_length=config['max_length'],
        system_instructions=args.system_instructions,
        prompt_template=args.prompt_template
    )
    
    # Handle validation split
    val_dataset = None
    if 'val' in dataset or 'validation' in dataset:
        val_split = 'val' if 'val' in dataset else 'validation'
        val_dataset = VLMDataset(
            dataset[val_split], processor, class_names,
            image_column=image_column, label_column=label_column,
            max_length=config['max_length'],
            system_instructions=args.system_instructions,
            prompt_template=args.prompt_template
        )
    elif 'test' in dataset:
        val_dataset = VLMDataset(
            dataset['test'], processor, class_names,
            image_column=image_column, label_column=label_column,
            max_length=config['max_length'],
            system_instructions=args.system_instructions,
            prompt_template=args.prompt_template
        )
    else:
        # Split training data
        train_val_split = dataset[train_split].train_test_split(test_size=args.val_split_ratio, seed=42)
        train_dataset = VLMDataset(
            train_val_split['train'], processor, class_names,
            image_column=image_column, label_column=label_column,
            max_length=config['max_length'],
            system_instructions=args.system_instructions,
            prompt_template=args.prompt_template
        )
        val_dataset = VLMDataset(
            train_val_split['test'], processor, class_names,
            image_column=image_column, label_column=label_column,
            max_length=config['max_length'],
            system_instructions=args.system_instructions,
            prompt_template=args.prompt_template
        )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=args.device != 'cpu', 
        collate_fn=lambda batch: vlm_collate_fn(batch, processor)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=args.device != 'cpu', 
        collate_fn=lambda batch: vlm_collate_fn(batch, processor)
    ) if val_dataset else None
    
    # Setup optimizer and scheduler
    # Only optimize unfrozen parameters (last layers + classification head)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
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
    logger.info("VLM training with frozen base model and trainable classification head")
    
    # Training loop
    best_val_accuracy = 0
    global_step = 0
    
    model.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # VLM forward pass
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=batch['class_labels']  # Use class labels for classification loss
            )
            loss = outputs['loss']
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'step': global_step
            })
            
            # Evaluate
            if global_step % args.eval_steps == 0 and val_dataloader:
                val_metrics = evaluate_vlm_model(model, val_dataloader, device, class_names)
                
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step} - Val Acc: {val_metrics['accuracy']:.4f}")
                
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    if accelerator.is_main_process:
                        # Save the VLM classifier
                        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_vlm_model.pth'))
                        logger.info(f"New best VLM model saved: {best_val_accuracy:.4f}")
                
                model.train()
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    if val_dataloader:
        final_metrics = evaluate_vlm_model(model, val_dataloader, device, class_names)
        if accelerator.is_main_process:
            logger.info("FINAL VLM RESULTS:")
            logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1: {final_metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1: {final_metrics['weighted_f1']:.4f}")
    
    logger.info(f"VLM training completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 