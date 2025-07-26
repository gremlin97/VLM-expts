#!/usr/bin/env python3
"""
Clean inference script for fine-tuned CLIP/SigLIP models on Mars terrain classification.
"""

import os
import json
import argparse
import logging
from typing import List, Dict

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, SiglipProcessor
from train import VisionEncoderClassifier, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarsClassifier:
    """Flexible Mars classifier using fine-tuned CLIP/SigLIP."""
    
    def __init__(self, model_path: str, model_key: str, class_names: List[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_key = model_key
        
        # Load class names from results file or use provided ones
        results_path = os.path.join(os.path.dirname(model_path), 'results.json')
        if class_names is None and os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                self.class_names = results.get('class_names', [])
                logger.info(f"Loaded class names from results: {self.class_names}")
        elif class_names is not None:
            self.class_names = class_names
        else:
            # Default Mars terrain classes
            self.class_names = [
                'ael', 'rou', 'cli', 'aec', 'tex', 'smo', 'fss', 'rid', 
                'fse', 'sfe', 'fsf', 'fsg', 'sfx', 'cra', 'mix'
            ]
            logger.warning("Using default Mars terrain class names")
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = MODEL_CONFIGS[model_key]
        
        # Load processor
        self.processor = config['processor_class'].from_pretrained(config['model_name'])
        
        # Load model
        self.model = VisionEncoderClassifier(
            model_key=model_key,
            num_classes=len(self.class_names),
            freeze_encoder=False
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized {model_key} classifier on {self.device}")
    
    def predict(self, image_path: str, return_probabilities: bool = False) -> Dict:
        """Predict terrain class for a single image."""
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(pixel_values)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        result = {
            'predicted_class': self.class_names[predicted_class_idx],
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict terrain classes for multiple images."""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_probabilities=True)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, save_path: str = None):
        """Visualize prediction with image and probabilities."""
        
        result = self.predict(image_path, return_probabilities=True)
        image = Image.open(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f"Model: {self.model_key}\n"
                     f"Predicted: {result['predicted_class']}\n"
                     f"Confidence: {result['confidence']:.3f}", fontsize=12)
        ax1.axis('off')
        
        # Show probabilities
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        classes = [classes[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]
        
        # Plot all classes
        y_pos = np.arange(len(classes))
        bars = ax2.barh(y_pos, probs, color='steelblue', alpha=0.7)
        
        # Highlight predicted class
        max_idx = np.argmax(probs)
        bars[max_idx].set_color('orange')
        bars[max_idx].set_alpha(1.0)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add probability values
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            if prob > 0.01:
                ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return result

def main():
    parser = argparse.ArgumentParser(description='Mars classification inference')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model weights (.pth file)')
    parser.add_argument('--model_key', type=str, required=True,
                      choices=list(MODEL_CONFIGS.keys()),
                      help='Model architecture key')
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                      help='List of class names (auto-loaded from results.json if available)')
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
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = MarsClassifier(args.model_path, args.model_key, args.class_names)
    
    if args.image_path:
        # Single image prediction
        logger.info(f"Processing image: {args.image_path}")
        
        result = classifier.predict(args.image_path, return_probabilities=True)
        
        print(f"\nPrediction Results:")
        print(f"Image: {args.image_path}")
        print(f"Model: {args.model_key}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if args.visualize:
            vis_output = args.vis_output or f"prediction_{args.model_key}_{os.path.basename(args.image_path)}.png"
            classifier.visualize_prediction(args.image_path, vis_output)
        
        # Show top 5 probabilities
        print(f"\nTop 5 Probabilities:")
        sorted_probs = sorted(
            result['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        for class_name, prob in sorted_probs:
            print(f"  {class_name}: {prob:.4f}")
    
    elif args.image_dir:
        # Batch prediction
        logger.info(f"Processing images in: {args.image_dir}")
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for file in os.listdir(args.image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.image_dir, file))
        
        if not image_paths:
            logger.error(f"No image files found in {args.image_dir}")
            return
        
        logger.info(f"Found {len(image_paths)} images")
        
        # Make predictions
        results = classifier.predict_batch(image_paths)
        
        # Print summary
        successful = [r for r in results if 'error' not in r]
        errors = [r for r in results if 'error' in r]
        
        print(f"\nBatch Results:")
        print(f"Total images: {len(image_paths)}")
        print(f"Successful: {len(successful)}")
        print(f"Errors: {len(errors)}")
        
        if successful:
            # Class distribution
            class_counts = {}
            for result in successful:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            print(f"\nClass Distribution:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}")
            
            # Overall statistics
            confidences = [r['confidence'] for r in successful]
            print(f"\nConfidence Statistics:")
            print(f"  Mean: {np.mean(confidences):.3f}")
            print(f"  Min: {np.min(confidences):.3f}")
            print(f"  Max: {np.max(confidences):.3f}")
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        
        # Show examples
        print(f"\nExample Predictions:")
        for result in successful[:5]:
            print(f"  {os.path.basename(result['image_path'])}: "
                  f"{result['predicted_class']} ({result['confidence']:.3f})")
    
    else:
        parser.error("Either --image_path or --image_dir must be specified")

if __name__ == "__main__":
    main() 