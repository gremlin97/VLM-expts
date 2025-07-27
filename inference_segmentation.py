#!/usr/bin/env python3
"""
Inference script for Mars segmentation using fine-tuned CLIP/SigLIP models.
"""

import argparse
import json
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from train_segmentation import VisionEncoderSegmentationHead, MODEL_CONFIGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarsSegmentationClassifier:
    """Mars segmentation classifier using fine-tuned CLIP/SigLIP."""
    
    def __init__(self, model_path: str, model_key: str, class_names: List[str] = None, target_size: int = 512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_key = model_key
        self.target_size = target_size
        
        # Load class names from results file or use provided ones
        results_path = os.path.join(os.path.dirname(model_path), 'results.json')
        if class_names is None and os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                self.class_names = results.get('class_names', [])
                self.target_size = results.get('target_size', 512)
                logger.info(f"Loaded class names from results: {self.class_names}")
                logger.info(f"Target size: {self.target_size}")
        elif class_names is not None:
            self.class_names = class_names
        else:
            # Default binary segmentation classes
            self.class_names = ['background', 'foreground']
            logger.warning("Using default binary segmentation class names")
        
        self.num_classes = len(self.class_names)
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = MODEL_CONFIGS[model_key]
        
        # Load processor
        self.processor = config['processor_class'].from_pretrained(config['model_name'])
        
        # Load model
        self.model = VisionEncoderSegmentationHead(
            model_key=model_key,
            num_classes=self.num_classes,
            target_size=self.target_size,
            freeze_encoder=False  # Inference model should not freeze encoder
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle both new checkpoint format and old format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info(f"Loaded model weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized {model_key} segmentation classifier on {self.device}")
    
    def predict(self, image_path: str) -> Dict:
        """Predict segmentation mask for a single image."""
        # Load and preprocess image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        # Resize to target size
        image_resized = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        
        # Process image
        inputs = self.processor(images=image_resized, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(pixel_values)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Convert to numpy
        probabilities_np = probabilities.cpu().numpy()[0]  # [num_classes, H, W]
        predictions_np = predictions.cpu().numpy()[0]  # [H, W]
        
        # Resize back to original size
        probabilities_resized = []
        for c in range(self.num_classes):
            prob_img = Image.fromarray((probabilities_np[c] * 255).astype(np.uint8))
            prob_resized = prob_img.resize(original_size, Image.BILINEAR)
            probabilities_resized.append(np.array(prob_resized) / 255.0)
        
        pred_img = Image.fromarray(predictions_np.astype(np.uint8))
        pred_resized = pred_img.resize(original_size, Image.NEAREST)
        predictions_resized = np.array(pred_resized)
        
        return {
            'predictions': predictions_resized,
            'probabilities': np.stack(probabilities_resized),
            'class_names': self.class_names,
            'original_size': original_size
        }
    
    def batch_predict(self, image_dir: str, output_file: str = None) -> List[Dict]:
        """Predict segmentation masks for all images in a directory."""
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = [f for f in os.listdir(image_dir) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        logger.info(f"Processing {len(image_files)} images from {image_dir}")
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            try:
                result = self.predict(image_path)
                result['image_file'] = image_file
                results.append(result)
                logger.info(f"Processed {image_file}")
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        # Save results if output file specified
        if output_file:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for result in results:
                json_result = {
                    'image_file': result['image_file'],
                    'class_names': result['class_names'],
                    'original_size': result['original_size'],
                    'predictions_shape': result['predictions'].shape,
                    'probabilities_shape': result['probabilities'].shape
                }
                json_results.append(json_result)
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def visualize_prediction(self, image_path: str, output_path: str = None, alpha: float = 0.5):
        """Visualize segmentation prediction with overlay."""
        result = self.predict(image_path)
        
        # Load original image
        original_image = Image.open(image_path)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        predictions = result['predictions']
        probabilities = result['probabilities']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Segmentation Results: {os.path.basename(image_path)}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Prediction mask
        axes[0, 1].imshow(predictions, cmap='tab10', vmin=0, vmax=len(self.class_names)-1)
        axes[0, 1].set_title('Predicted Segmentation')
        axes[0, 1].axis('off')
        
        # Add colorbar for segmentation
        cbar = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar.set_ticks(range(len(self.class_names)))
        cbar.set_ticklabels(self.class_names)
        
        # Overlay
        overlay = np.array(original_image)
        
        # Create colored mask for overlay
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        colored_mask = np.zeros((*predictions.shape, 3))
        for i in range(len(self.class_names)):
            mask = predictions == i
            colored_mask[mask] = colors[i][:3]
        
        # Blend with original image
        blended = (1 - alpha) * (overlay / 255.0) + alpha * colored_mask
        axes[1, 0].imshow(blended)
        axes[1, 0].set_title(f'Overlay (alpha={alpha})')
        axes[1, 0].axis('off')
        
        # Probability heatmap for foreground class (if binary)
        if len(self.class_names) == 2:
            prob_map = probabilities[1]  # Foreground probabilities
            im = axes[1, 1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
            axes[1, 1].set_title(f'{self.class_names[1]} Probability')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        else:
            # Show class distribution
            class_counts = np.bincount(predictions.flatten(), minlength=len(self.class_names))
            class_percentages = class_counts / predictions.size * 100
            
            bars = axes[1, 1].bar(self.class_names, class_percentages)
            axes[1, 1].set_title('Class Distribution (%)')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, class_percentages):
                if pct > 0:
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Mars segmentation inference')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model weights (.pth file)')
    parser.add_argument('--model_key', type=str, required=True,
                      choices=list(MODEL_CONFIGS.keys()),
                      help='Model architecture key')
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                      help='List of class names (auto-loaded from results.json if available)')
    parser.add_argument('--target_size', type=int, default=512,
                      help='Target size for inference (auto-loaded from results.json if available)')
    parser.add_argument('--image_path', type=str,
                      help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                      help='Directory containing images for batch prediction')
    parser.add_argument('--output_file', type=str,
                      help='Output file for batch predictions (JSON)')
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualization of prediction')
    parser.add_argument('--vis_output', type=str,
                      help='Output path for visualization')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Alpha value for overlay visualization')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        parser.error("Either --image_path or --image_dir must be specified")
    
    # Initialize classifier
    classifier = MarsSegmentationClassifier(args.model_path, args.model_key, args.class_names, args.target_size)
    
    if args.image_path:
        # Single image prediction
        if args.visualize:
            result = classifier.visualize_prediction(args.image_path, args.vis_output, args.alpha)
        else:
            result = classifier.predict(args.image_path)
        
        logger.info(f"Prediction completed for {args.image_path}")
        logger.info(f"Image size: {result['original_size']}")
        logger.info(f"Classes: {result['class_names']}")
        
        # Print class distribution
        predictions = result['predictions']
        class_counts = np.bincount(predictions.flatten(), minlength=len(result['class_names']))
        class_percentages = class_counts / predictions.size * 100
        
        for class_name, percentage in zip(result['class_names'], class_percentages):
            logger.info(f"  {class_name}: {percentage:.2f}%")
    
    elif args.image_dir:
        # Batch prediction
        results = classifier.batch_predict(args.image_dir, args.output_file)
        
        logger.info(f"Batch prediction completed for {len(results)} images")
        
        # Print summary statistics
        if results:
            all_predictions = []
            for result in results:
                all_predictions.extend(result['predictions'].flatten())
            
            overall_counts = np.bincount(all_predictions, minlength=len(classifier.class_names))
            overall_percentages = overall_counts / len(all_predictions) * 100
            
            logger.info("Overall class distribution:")
            for class_name, percentage in zip(classifier.class_names, overall_percentages):
                logger.info(f"  {class_name}: {percentage:.2f}%")

if __name__ == "__main__":
    main() 