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
    """Mars terrain dataset."""
    
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Ensure RGB format
        if image.mode != 'RGB':
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
    parser = argparse.ArgumentParser(description='Fine-tune CLIP/SigLIP image encoders on Mars dataset')
    parser.add_argument('--model', type=str, 
                      choices=list(MODEL_CONFIGS.keys()),
                      default='clip-vit-base-patch32',
                      help='Model to fine-tune')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--freeze_encoder', action='store_true', 
                      help='Freeze vision encoder weights')
    parser.add_argument('--output_dir', type=str, default='./clip_results', 
                      help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation frequency')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator()
    device = accelerator.device
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project='mars-clip-siglip', config=vars(args))
    
    # Mars terrain classes
    class_names = [
        'ael', 'rou', 'cli', 'aec', 'tex', 'smo', 'fss', 'rid', 
        'fse', 'sfe', 'fsf', 'fsg', 'sfx', 'cra', 'mix'
    ]
    
    logger.info(f"Fine-tuning {args.model} on Mars terrain classification")
    logger.info(f"Classes: {len(class_names)}")
    logger.info(f"Freeze encoder: {args.freeze_encoder}")
    
    # Load dataset
    logger.info("Loading Mars dataset...")
    dataset = load_dataset("Mirali33/mb-domars16k")
    
    # Initialize processor and model
    config = MODEL_CONFIGS[args.model]
    processor = config['processor_class'].from_pretrained(config['model_name'])
    model = VisionEncoderClassifier(
        model_key=args.model,
        num_classes=len(class_names),
        freeze_encoder=args.freeze_encoder
    )
    
    # Create datasets
    train_dataset = MarsDataset(dataset['train'], processor)
    val_dataset = MarsDataset(dataset['val'], processor)
    test_dataset = MarsDataset(dataset['test'], processor)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
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
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
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
            if global_step % args.eval_steps == 0:
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
                        torch.save(
                            accelerator.unwrap_model(model).state_dict(),
                            os.path.join(args.output_dir, 'best_model.pth')
                        )
                        logger.info(f"New best model saved: {best_val_accuracy:.4f}")
                
                model.train()
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation
    logger.info("Final evaluation on test set...")
    
    # Load best model
    if accelerator.is_main_process:
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    test_metrics = evaluate_model(model, test_dataloader, device, class_names)
    
    if accelerator.is_main_process:
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        
        # Save results
        results = {
            'model': args.model,
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'best_val_accuracy': best_val_accuracy,
            'args': vars(args)
        }
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        if train_losses and val_accuracies:
            plot_training_curves(
                train_losses, val_accuracies,
                os.path.join(args.output_dir, 'training_curves.png')
            )
        
        # Save final model
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(args.output_dir, 'final_model.pth')
        )
        
        if args.use_wandb:
            wandb.log({
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1'],
                'test_weighted_f1': test_metrics['weighted_f1']
            })
            wandb.finish()
        
        logger.info(f"Training completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 