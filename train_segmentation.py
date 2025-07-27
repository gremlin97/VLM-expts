#!/usr/bin/env python3
"""
Fine-tune CLIP/SigLIP image encoders for Mars segmentation tasks.
Extends the classification framework to handle semantic segmentation.
"""

import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    SiglipVisionModel, SiglipImageProcessor,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
from sklearn.metrics import accuracy_score, jaccard_score
import wandb
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'clip-vit-base-patch32': {
        'model_name': 'openai/clip-vit-base-patch32',
        'vision_model_class': CLIPVisionModel,
        'processor_class': CLIPImageProcessor,
        'hidden_size': 768,
        'patch_size': 32
    },
    'clip-vit-base-patch16': {
        'model_name': 'openai/clip-vit-base-patch16',
        'vision_model_class': CLIPVisionModel,
        'processor_class': CLIPImageProcessor,
        'hidden_size': 768,
        'patch_size': 16
    },
    'clip-vit-large-patch14': {
        'model_name': 'openai/clip-vit-large-patch14',
        'vision_model_class': CLIPVisionModel,
        'processor_class': CLIPImageProcessor,
        'hidden_size': 1024,
        'patch_size': 14
    },
    'siglip-base-patch16-224': {
        'model_name': 'google/siglip-base-patch16-224',
        'vision_model_class': SiglipVisionModel,
        'processor_class': SiglipImageProcessor,
        'hidden_size': 768,
        'patch_size': 16
    },
    'siglip-large-patch16-256': {
        'model_name': 'google/siglip-large-patch16-256',
        'vision_model_class': SiglipVisionModel,
        'processor_class': SiglipImageProcessor,
        'hidden_size': 1024,
        'patch_size': 16
    }
}

class MarsSegmentationDataset(torch.utils.data.Dataset):
    """Mars segmentation dataset for various segmentation tasks."""
    
    def __init__(self, dataset, processor, image_column='image', mask_column='mask', target_size=512):
        self.dataset = dataset
        self.processor = processor
        self.image_column = image_column
        self.mask_column = mask_column
        self.target_size = target_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        mask = item[self.mask_column]
        
        # Ensure RGB format for image
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert mask to grayscale if needed
        if hasattr(mask, 'mode') and mask.mode != 'L':
            mask = mask.convert('L')
        
        # Resize both image and mask to target size
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Convert mask to tensor
        mask_array = np.array(mask)
        mask_tensor = torch.from_numpy(mask_array).long()
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': mask_tensor
        }

class VisionEncoderSegmentationHead(nn.Module):
    """Vision encoder with segmentation decoder head."""
    
    def __init__(self, model_key: str, num_classes: int = 2, target_size: int = 512, freeze_encoder: bool = False):
        super().__init__()
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = MODEL_CONFIGS[model_key]
        self.patch_size = config['patch_size']
        self.target_size = target_size
        
        # Calculate feature map dimensions
        self.feature_size = target_size // self.patch_size
        
        # Load vision encoder
        self.vision_encoder = config['vision_model_class'].from_pretrained(config['model_name'])
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")
        
        # Segmentation decoder
        hidden_size = config['hidden_size']
        
        # Decoder layers to upsample from patch features to full resolution
        self.decoder = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(hidden_size, hidden_size // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(inplace=True),
            
            # Second upsampling block
            nn.ConvTranspose2d(hidden_size // 2, hidden_size // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(inplace=True),
            
            # Third upsampling block (if needed for patch size 32)
            nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size // 8),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(hidden_size // 8, num_classes, kernel_size=3, padding=1)
        )
        
        # Additional upsampling if patch size is 32
        if self.patch_size == 32:
            self.final_upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        else:
            self.final_upsample = None
        
        logger.info(f"Loaded {model_key} segmentation model with {hidden_size}D features")
        logger.info(f"Feature map size: {self.feature_size}x{self.feature_size}")
    
    def forward(self, pixel_values):
        # Get vision features
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        
        # Get the last hidden state (patch embeddings)
        patch_embeddings = vision_outputs.last_hidden_state  # [B, num_patches+1, hidden_size]
        
        # Remove CLS token (first token)
        patch_embeddings = patch_embeddings[:, 1:, :]  # [B, num_patches, hidden_size]
        
        # Reshape to spatial feature map
        batch_size = patch_embeddings.shape[0]
        num_patches = patch_embeddings.shape[1]
        hidden_size = patch_embeddings.shape[2]
        
        # Calculate actual feature size from number of patches
        actual_feature_size = int(np.sqrt(num_patches))
        
        # Debug: Check if we have a perfect square
        if actual_feature_size * actual_feature_size != num_patches:
            # If not perfect square, find the closest square
            actual_feature_size = int(np.sqrt(num_patches))
            # Pad or truncate to make it work
            if actual_feature_size * actual_feature_size < num_patches:
                actual_feature_size += 1
            
            # Reshape to the closest square
            target_patches = actual_feature_size * actual_feature_size
            if num_patches < target_patches:
                # Pad with zeros
                padding = torch.zeros(batch_size, target_patches - num_patches, hidden_size, device=patch_embeddings.device)
                patch_embeddings = torch.cat([patch_embeddings, padding], dim=1)
            else:
                # Truncate
                patch_embeddings = patch_embeddings[:, :target_patches, :]
        
        patch_embeddings = patch_embeddings.transpose(1, 2)  # [B, hidden_size, num_patches]
        feature_maps = patch_embeddings.view(batch_size, hidden_size, actual_feature_size, actual_feature_size)
        
        # Apply decoder
        output = self.decoder(feature_maps)
        
        # Final upsampling if needed
        if self.final_upsample is not None:
            output = self.final_upsample(output)
        
        # Resize to target size if needed
        if output.shape[-1] != self.target_size:
            output = F.interpolate(output, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        
        return output

def compute_segmentation_metrics(predictions, labels, num_classes):
    """Compute segmentation metrics."""
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Get predicted classes
    pred_classes = np.argmax(predictions, axis=1)
    
    # Flatten for metric computation
    pred_flat = pred_classes.flatten()
    labels_flat = labels.flatten()
    
    # Compute metrics
    accuracy = accuracy_score(labels_flat, pred_flat)
    
    # Compute IoU (Jaccard score) for each class
    iou_scores = []
    for class_id in range(num_classes):
        class_pred = (pred_flat == class_id)
        class_true = (labels_flat == class_id)
        
        if class_true.sum() > 0:  # Only compute if class exists in ground truth
            iou = jaccard_score(class_true, class_pred, average=None)
            # jaccard_score with average=None returns an array, we want the first (and only) element
            if isinstance(iou, np.ndarray):
                iou = float(iou[0])
            else:
                iou = float(iou)
            iou_scores.append(iou)
        else:
            iou_scores.append(0.0)
    
    mean_iou = np.mean(iou_scores)
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'class_iou': iou_scores
    }

def evaluate_model(model, dataloader, device, num_classes):
    """Evaluate the segmentation model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.append(outputs)
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_segmentation_metrics(all_predictions, all_labels, num_classes)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def plot_training_curves(train_losses, val_metrics, output_path):
    """Plot training curves for segmentation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Validation loss
    val_losses = [m['loss'] for m in val_metrics]
    ax2.plot(val_losses)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # Validation accuracy
    val_accuracies = [m['accuracy'] for m in val_metrics]
    ax3.plot(val_accuracies)
    ax3.set_title('Validation Accuracy')
    ax3.set_xlabel('Evaluation Step')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True)
    
    # Validation mIoU
    val_mious = [m['mean_iou'] for m in val_metrics]
    ax4.plot(val_mious)
    ax4.set_title('Validation Mean IoU')
    ax4.set_xlabel('Evaluation Step')
    ax4.set_ylabel('Mean IoU')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP/SigLIP image encoders for Mars segmentation')
    parser.add_argument('--model', type=str, 
                      choices=list(MODEL_CONFIGS.keys()),
                      default='clip-vit-base-patch32',
                      help='Model to fine-tune')
    parser.add_argument('--dataset', type=str, default='Mirali33/mb-crater_binary_seg',
                      help='HuggingFace dataset name')
    parser.add_argument('--dataset_config', type=str, default=None,
                      help='Dataset configuration name')
    parser.add_argument('--image_column', type=str, default='image',
                      help='Name of image column in dataset')
    parser.add_argument('--mask_column', type=str, default='mask',
                      help='Name of mask column in dataset')
    parser.add_argument('--num_classes', type=int, default=None,
                      help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--target_size', type=int, default=512,
                      help='Target image/mask size for training')
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                      help='List of class names (optional)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--freeze_encoder', action='store_true', 
                      help='Freeze vision encoder weights')
    parser.add_argument('--output_dir', type=str, default='./segmentation_results', 
                      help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use from each split (for testing)')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto/cuda/mps/cpu). auto will use GPU if available')
    
    args = parser.parse_args()
    
    # Create output directory
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
        wandb.init(project='mars-segmentation-clip-siglip', config=vars(args))
    
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
    
    # Auto-detect number of classes from class_labels if available
    if args.num_classes is None:
        if 'class_labels' in dataset['train'].features:
            # Get class labels from the first sample
            sample_labels = dataset['train'][0]['class_labels']
            args.num_classes = len(sample_labels)
            logger.info(f"Auto-detected {args.num_classes} classes from class_labels")
        else:
            # For binary segmentation, assume 2 classes (background + foreground)
            args.num_classes = 2
            logger.info(f"Assuming binary segmentation with {args.num_classes} classes")
    
    # Set class names
    if args.class_names is None:
        if 'class_labels' in dataset['train'].features:
            class_names = dataset['train'][0]['class_labels']
        elif args.num_classes == 2:
            class_names = ['background', 'foreground']
        else:
            class_names = [f'class_{i}' for i in range(args.num_classes)]
    else:
        class_names = args.class_names
        if len(class_names) != args.num_classes:
            raise ValueError(f"Number of class names ({len(class_names)}) doesn't match num_classes ({args.num_classes})")
    
    logger.info(f"Fine-tuning {args.model} on {args.dataset}")
    logger.info(f"Classes: {args.num_classes} - {class_names}")
    logger.info(f"Target size: {args.target_size}x{args.target_size}")
    logger.info(f"Freeze encoder: {args.freeze_encoder}")
    
    # Initialize processor and model
    config = MODEL_CONFIGS[args.model]
    processor = config['processor_class'].from_pretrained(config['model_name'])
    
    model = VisionEncoderSegmentationHead(
        model_key=args.model,
        num_classes=args.num_classes,
        target_size=args.target_size,
        freeze_encoder=args.freeze_encoder
    )
    
    # Handle different dataset splits
    val_dataset = None
    test_dataset = None
    
    if 'val' in dataset or 'validation' in dataset:
        val_split = 'val' if 'val' in dataset else 'validation'
        train_dataset = MarsSegmentationDataset(dataset['train'], processor, args.image_column, args.mask_column, args.target_size)
        val_dataset = MarsSegmentationDataset(dataset[val_split], processor, args.image_column, args.mask_column, args.target_size)
    elif 'test' in dataset:
        # Use test as validation if no val split
        train_dataset = MarsSegmentationDataset(dataset['train'], processor, args.image_column, args.mask_column, args.target_size)
        val_dataset = MarsSegmentationDataset(dataset['test'], processor, args.image_column, args.mask_column, args.target_size)
    else:
        # Split training data if no validation set
        logger.info("No validation set found, splitting training data 80/20")
        train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_dataset = MarsSegmentationDataset(train_val_split['train'], processor, args.image_column, args.mask_column, args.target_size)
        val_dataset = MarsSegmentationDataset(train_val_split['test'], processor, args.image_column, args.mask_column, args.target_size)
    
    if 'test' in dataset and ('val' in dataset or 'validation' in dataset):
        test_dataset = MarsSegmentationDataset(dataset['test'], processor, args.image_column, args.mask_column, args.target_size)
    
    val_len = len(val_dataset) if val_dataset else 0
    test_len = len(test_dataset) if test_dataset else 0
    logger.info(f"Train: {len(train_dataset)}, Val: {val_len}, Test: {test_len}")
    
    # Create dataloaders with appropriate settings for device
    num_workers = 2 if args.device not in ['cpu'] else 0
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
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
    model.train()
    global_step = 0
    train_losses = []
    val_metrics_history = []
    best_val_miou = 0.0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            global_step += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Evaluate
            if global_step % args.eval_steps == 0 and val_dataloader:
                val_metrics = evaluate_model(model, val_dataloader, device, args.num_classes)
                val_metrics_history.append(val_metrics)
                
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step} - Val Acc: {val_metrics['accuracy']:.4f}, "
                              f"Val mIoU: {val_metrics['mean_iou']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                
                if args.use_wandb and accelerator.is_main_process:
                    wandb.log({
                        'val_accuracy': val_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_mean_iou': val_metrics['mean_iou'],
                        'global_step': global_step
                    })
                
                # Save best model
                if val_metrics['mean_iou'] > best_val_miou:
                    best_val_miou = val_metrics['mean_iou']
                    if accelerator.is_main_process:
                        checkpoint = {
                            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                            'model_architecture': args.model,
                            'best_val_miou': best_val_miou,
                            'args': vars(args)
                        }
                        torch.save(
                            checkpoint,
                            os.path.join(args.output_dir, 'best_model.pth'),
                            _use_new_zipfile_serialization=False
                        )
                        logger.info(f"New best model saved: mIoU {best_val_miou:.4f}")
                
                model.train()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_epoch_loss:.4f}")
    
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
        
        test_metrics = evaluate_model(model, test_dataloader, device, args.num_classes)
    elif val_dataloader:
        logger.info("No test set available, using validation set for final evaluation...")
        test_metrics = evaluate_model(model, val_dataloader, device, args.num_classes)
    else:
        logger.info("No test or validation set available, skipping final evaluation")
    
    if accelerator.is_main_process:
        if test_metrics:
            logger.info(f"Final Results:")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Mean IoU: {test_metrics['mean_iou']:.4f}")
            logger.info(f"  Class IoU: {test_metrics['class_iou']}")
        
        # Save results
        results = {
            'model': args.model,
            'dataset': args.dataset,
            'num_classes': args.num_classes,
            'class_names': class_names,
            'target_size': args.target_size,
            'best_val_miou': best_val_miou,
            'args': vars(args)
        }

        if test_metrics:
            results.update({
                'test_accuracy': test_metrics['accuracy'],
                'test_mean_iou': test_metrics['mean_iou'],
                'test_class_iou': [float(iou) for iou in test_metrics['class_iou']]
            })
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        if train_losses and val_metrics_history:
            plot_training_curves(
                train_losses, val_metrics_history,
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
                'test_mean_iou': test_metrics['mean_iou']
            })
            wandb.finish()
        
        logger.info(f"Training completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 