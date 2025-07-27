#!/usr/bin/env python3
"""
Clean implementation for fine-tuning CLIP and SigLIP image encoders on Mars terrain classification.
"""

import os
import json
import argparse
import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    CLIPVisionModel, CLIPProcessor,
    SiglipVisionModel, SiglipProcessor,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'clip-vit-base-patch32': {
        'vision_model_class': CLIPVisionModel,
        'processor_class': CLIPProcessor,
        'model_name': 'openai/clip-vit-base-patch32',
        'hidden_size': 768
    },
    'clip-vit-base-patch16': {
        'vision_model_class': CLIPVisionModel,
        'processor_class': CLIPProcessor,
        'model_name': 'openai/clip-vit-base-patch16',
        'hidden_size': 768
    },
    'clip-vit-large-patch14': {
        'vision_model_class': CLIPVisionModel,
        'processor_class': CLIPProcessor,
        'model_name': 'openai/clip-vit-large-patch14',
        'hidden_size': 1024
    },
    'siglip-base-patch16-224': {
        'vision_model_class': SiglipVisionModel,
        'processor_class': SiglipProcessor,
        'model_name': 'google/siglip-base-patch16-224',
        'hidden_size': 768
    },
    'siglip-large-patch16-256': {
        'vision_model_class': SiglipVisionModel,
        'processor_class': SiglipProcessor,
        'model_name': 'google/siglip-large-patch16-256',
        'hidden_size': 1024
    }
}

class MarsDataset(torch.utils.data.Dataset):
    """Flexible Mars dataset for various configurations."""
    
    def __init__(self, dataset, processor, image_column='image', label_column='label'):
        self.dataset = dataset
        self.processor = processor
        self.image_column = image_column
        self.label_column = label_column
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        label = item[self.label_column]
        
        # Ensure RGB format
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

class VisionEncoderClassifier(nn.Module):
    """Vision encoder with classification head."""
    
    def __init__(self, model_key: str, num_classes: int = 15, freeze_encoder: bool = False):
        super().__init__()
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = MODEL_CONFIGS[model_key]
        
        # Load vision encoder
        self.vision_encoder = config['vision_model_class'].from_pretrained(config['model_name'])
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")
        
        # Classification head
        hidden_size = config['hidden_size']
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        logger.info(f"Loaded {model_key} with {hidden_size}D features")
        
    def forward(self, pixel_values):
        # Get vision features
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        pooled_features = vision_outputs.pooler_output
        
        # Classify
        logits = self.classifier(pooled_features)
        return logits

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
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
        'loss': total_loss / len(dataloader),
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'classification_report': report
    }

def plot_training_curves(train_losses, val_accuracies, save_path):
    """Plot training curves."""
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

def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP/SigLIP image encoders on Mars datasets')
    parser.add_argument('--model', type=str, 
                      choices=list(MODEL_CONFIGS.keys()),
                      default='clip-vit-base-patch32',
                      help='Model to fine-tune')
    parser.add_argument('--dataset', type=str, default='Mirali33/mb-domars16k',
                      help='HuggingFace dataset name')
    parser.add_argument('--dataset_config', type=str, default=None,
                      help='Dataset configuration name')
    parser.add_argument('--image_column', type=str, default='image',
                      help='Name of image column in dataset')
    parser.add_argument('--label_column', type=str, default='label',
                      help='Name of label column in dataset')
    parser.add_argument('--num_classes', type=int, default=None,
                      help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                      help='List of class names (optional)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--freeze_encoder', action='store_true', 
                      help='Freeze vision encoder weights')
    parser.add_argument('--output_dir', type=str, default='./results', 
                      help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use from each split (for testing)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto/cuda/mps/cpu). auto will use GPU if available')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project='mars-clip-siglip', config=vars(args))
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset)
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None:
        logger.info(f"Limiting each split to {args.max_samples} samples for testing")
        for split in dataset.keys():
            if len(dataset[split]) > args.max_samples:
                dataset[split] = dataset[split].select(range(args.max_samples))
                logger.info(f"Limited {split} split to {len(dataset[split])} samples")
    
    # Auto-detect number of classes and class names
    if args.num_classes is None:
        # Get unique labels from training set
        train_labels = set()
        for item in dataset['train']:
            train_labels.add(item[args.label_column])
        args.num_classes = len(train_labels)
        logger.info(f"Auto-detected {args.num_classes} classes")
    
    # Set class names
    if args.class_names is None:
        if args.num_classes == 2:
            class_names = ['negative', 'positive']
        elif args.dataset == 'Mirali33/mb-domars16k':
            # Default Mars terrain classes
            class_names = [
                'ael', 'rou', 'cli', 'aec', 'tex', 'smo', 'fss', 'rid', 
                'fse', 'sfe', 'fsf', 'fsg', 'sfx', 'cra', 'mix'
            ]
        else:
            # Generic class names
            class_names = [f'class_{i}' for i in range(args.num_classes)]
    else:
        class_names = args.class_names
        if len(class_names) != args.num_classes:
            raise ValueError(f"Number of class names ({len(class_names)}) doesn't match num_classes ({args.num_classes})")
    
    logger.info(f"Fine-tuning {args.model} on {args.dataset}")
    logger.info(f"Classes: {args.num_classes} - {class_names}")
    logger.info(f"Freeze encoder: {args.freeze_encoder}")
    
    # Initialize processor and model
    config = MODEL_CONFIGS[args.model]
    processor = config['processor_class'].from_pretrained(config['model_name'])
    model = VisionEncoderClassifier(
        model_key=args.model,
        num_classes=args.num_classes,
        freeze_encoder=args.freeze_encoder
    )
    
    # Handle different dataset splits
    val_dataset = None
    test_dataset = None
    
    if 'val' in dataset or 'validation' in dataset:
        val_split = 'val' if 'val' in dataset else 'validation'
        train_dataset = MarsDataset(dataset['train'], processor, args.image_column, args.label_column)
        val_dataset = MarsDataset(dataset[val_split], processor, args.image_column, args.label_column)
    elif 'test' in dataset:
        # Use test as validation if no val split
        train_dataset = MarsDataset(dataset['train'], processor, args.image_column, args.label_column)
        val_dataset = MarsDataset(dataset['test'], processor, args.image_column, args.label_column)
    else:
        # Split training data if no validation set
        logger.info("No validation set found, splitting training data 80/20")
        train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_dataset = MarsDataset(train_val_split['train'], processor, args.image_column, args.label_column)
        val_dataset = MarsDataset(train_val_split['test'], processor, args.image_column, args.label_column)
    
    if 'test' in dataset and ('val' in dataset or 'validation' in dataset):
        test_dataset = MarsDataset(dataset['test'], processor, args.image_column, args.label_column)
    
    val_len = len(val_dataset) if val_dataset else 0
    test_len = len(test_dataset) if test_dataset else 0
    logger.info(f"Train: {len(train_dataset)}, Val: {val_len}, Test: {test_len}")
    
    # Create dataloaders with appropriate settings for device
    num_workers = 4 if args.device not in ['cpu'] else 0
    pin_memory = args.device not in ['cpu']
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    ) if val_dataset else None
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
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
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            
            # Forward pass
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
            
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
                val_metrics = evaluate_model(model, val_dataloader, device, class_names)
                val_accuracies.append(val_metrics['accuracy'])
                
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step} - Val Acc: {val_metrics['accuracy']:.4f}, "
                              f"Val Loss: {val_metrics['loss']:.4f}")
                
                if args.use_wandb and accelerator.is_main_process:
                    wandb.log({
                        'val_accuracy': val_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_macro_f1': val_metrics['macro_f1'],
                        'global_step': global_step
                    })
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    if accelerator.is_main_process:
                        checkpoint = {
                            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                            'model_architecture': args.model,
                            'best_val_accuracy': best_val_accuracy,
                            'args': vars(args)
                        }
                        torch.save(
                            checkpoint,
                            os.path.join(args.output_dir, 'best_model.pth'),
                            _use_new_zipfile_serialization=False
                        )
                        logger.info(f"New best model saved: {best_val_accuracy:.4f}")
                
                model.train()
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    test_metrics = None
    if test_dataloader:
        logger.info("Final evaluation on test set...")
        
        # Load best model
        if accelerator.is_main_process:
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                try:
                    # Check if the checkpoint is compatible with current model
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    
                    # Check if this checkpoint was saved with the same model architecture
                    if 'model_architecture' in checkpoint:
                        if checkpoint['model_architecture'] == args.model:
                            model.load_state_dict(checkpoint['model_state_dict'])
                            logger.info(f"Successfully loaded best model from {best_model_path}")
                        else:
                            logger.warning(f"Checkpoint was saved with {checkpoint['model_architecture']}, but current model is {args.model}")
                            logger.info("Continuing with current model state for evaluation")
                    else:
                        # Try to load anyway (for backward compatibility)
                        model.load_state_dict(checkpoint)
                        logger.info(f"Successfully loaded best model from {best_model_path}")
                        
                except RuntimeError as e:
                    logger.warning(f"Could not load best model due to architecture mismatch: {e}")
                    logger.info("Continuing with current model state for evaluation")
        
        test_metrics = evaluate_model(model, test_dataloader, device, class_names)
    elif val_dataloader:
        logger.info("No test set available, using validation set for final evaluation...")
        test_metrics = evaluate_model(model, val_dataloader, device, class_names)
    else:
        logger.info("No test or validation set available, skipping final evaluation")
    
    if accelerator.is_main_process:
        if test_metrics:
            logger.info(f"Final Results:")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        
        # Save results
        results = {
            'model': args.model,
            'dataset': args.dataset,
            'num_classes': args.num_classes,
            'class_names': class_names,
            'best_val_accuracy': best_val_accuracy,
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
        if train_losses and val_accuracies:
            plot_training_curves(
                train_losses, val_accuracies,
                os.path.join(args.output_dir, 'training_curves.png')
            )
        
        # Save final model
        final_checkpoint = {
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'model_architecture': args.model,
            'args': vars(args)
        }
        torch.save(
            final_checkpoint,
            os.path.join(args.output_dir, 'final_model.pth'),
            _use_new_zipfile_serialization=False
        )
        
        if args.use_wandb and test_metrics:
            wandb.log({
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1'],
                'test_weighted_f1': test_metrics['weighted_f1']
            })
            wandb.finish()
        
        logger.info(f"Training completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 