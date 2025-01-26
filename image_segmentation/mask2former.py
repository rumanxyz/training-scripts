"""
Mask2Former Semantic Segmentation Training Pipeline (Hugging Face Transformers)
"""

import cv2
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Hugging Face Imports
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

def create_segmentation_mask(image_shape: Tuple[int, int], polygons: List[Dict], num_classes: int) -> torch.Tensor:
    """
    Create binary mask tensor for Mask2Former (one-hot encoded per class).
    
    Args:
        image_shape: (height, width) of the mask
        polygons: List of polygon annotations
        num_classes: Total number of classes including background
    
    Returns:
        Tensor of shape (num_classes, height, width)
    """
    mask = np.zeros((num_classes, image_shape[0], image_shape[1]), dtype=np.uint8)
    for polygon in polygons:
        class_id = polygon["class"]
        points = np.array([[int(cord[0]*image_shape[1]), int(cord[1]*image_shape[0])] 
                         for cord in polygon["points"]], dtype=np.int32)
        cv2.fillPoly(mask[class_id], [points], color=1)
    return torch.from_numpy(mask).float()

class CustomSegmentationDataset(Dataset):
    """
    Mask2Former-compatible dataset with size validation and one-hot encoded masks.
    """
    def __init__(self, csv_file_path: str, image_size: Tuple[int, int], num_classes: int, 
                 transform: Optional[transforms.Compose] = None):
        # Validate image size requirements
        if image_size[0] % 32 != 0 or image_size[1] % 32 != 0:
            raise ValueError("Image dimensions must be divisible by 32 (e.g., 512x512)")
            
        self.df_data = pd.read_csv(csv_file_path)
        self.image_files = self.df_data.image_path.values
        self.image_annotations = self.df_data.label.values
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load and resize image
        image = Image.open(self.image_files[idx]).convert('RGB')
        image = image.resize(self.image_size)
        
        # Load and process annotations
        with open(self.image_annotations[idx], 'r') as f:
            polygons = json.load(f)
        
        # Create one-hot encoded mask
        mask = create_segmentation_mask(image.size[::-1], polygons, self.num_classes)
        
        if self.transform:
            image = self.transform(image)
            
        return image, mask

def load_pretrained_mask2former(num_classes: int) -> nn.Module:
    """Load Mask2Former model with proper configuration"""
    config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-small-coco-instance")
    config.num_labels = num_classes
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance",
        config=config,
        ignore_mismatched_sizes=True
    )

def calculate_metrics(predictions, targets, num_classes, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """Convert query-based outputs to semantic masks before metric calculation"""
    # Convert model outputs to segmentation masks
    class_queries = predictions.class_queries_logits  # (B, num_queries, num_classes)
    mask_queries = predictions.masks_queries_logits  # (B, num_queries, H, W)
    
    # Combine queries to create final mask
    pred_masks = torch.einsum("bqc, bqhw -> bchw", class_queries.softmax(dim=-1), mask_queries.sigmoid())
    pred_masks = pred_masks.argmax(dim=1)  # (B, H, W)

    # Calculate metrics
    metrics = {
        "accuracy": (pred_masks == targets).float().mean().item(),
        "class_metrics": {}
    }
    
    for cls in range(num_classes):
        cls_pred = (pred_masks == cls)
        cls_target = (targets == cls)
        
        intersection = (cls_pred & cls_target).float().sum()
        union = (cls_pred | cls_target).float().sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        metrics["class_metrics"][cls] = {
            "iou": iou.item(),
            "threshold_accuracy": {}
        }
        
        for thresh in thresholds:
            metrics["class_metrics"][cls]["threshold_accuracy"][f"{int(thresh*100)}%"] = (
                (iou >= thresh).float().mean().item()
            )
    
    return metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Union[str, Path],
    learning_rate: float,
    num_classes: int,
    num_epochs: int = 10,
    save_interval: int = 5
) -> None:
    """Training loop adapted for Mask2Former's output structure"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_iou = 0.0
    best_model_path = save_dir / "best_mask2former.pth"
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images, mask_labels=masks)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(pixel_values=images, mask_labels=masks)
                val_loss += outputs.loss.item()
                
                # Store for metric calculation
                all_preds.append(outputs)
                all_targets.append(masks)
        
        # Calculate metrics
        val_preds = type(outputs)(*map(torch.cat, zip(*all_preds)))
        val_targets = torch.cat(all_targets)
        val_metrics = calculate_metrics(val_preds, val_targets, num_classes)
        
        # Save best model
        current_iou = np.mean([v["iou"] for v in val_metrics["class_metrics"].values()])
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes
            }, best_model_path)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Mean IoU: {current_iou:.4f}")
        print("Class-wise IoUs:")
        for cls, metrics in val_metrics["class_metrics"].items():
            print(f"Class {cls}: {metrics['iou']:.4f}")
        
        scheduler.step()

def main():
    parser = argparse.ArgumentParser(description="Train Mask2Former for Semantic Segmentation")
    parser.add_argument('--train_csv_path', type=str, required=True,
                      help="Path to training CSV with image_path and label columns")
    parser.add_argument('--val_csv_path', type=str, required=True,
                      help="Path to validation CSV with image_path and label columns")
    parser.add_argument('--num_classes', type=int, required=True,
                      help="Number of classes including background")
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                      help='Image size (width, height) must be divisible by 32')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--save_dir', type=str, default="checkpoints")
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.num_classes < 2:
        raise ValueError("num_classes must be at least 2 (background + at least one class)")
    if any(s % 32 != 0 for s in args.image_size):
        raise ValueError("Image dimensions must be divisible by 32")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CustomSegmentationDataset(
        args.train_csv_path,
        tuple(args.image_size),
        args.num_classes,
        transform
    )
    val_dataset = CustomSegmentationDataset(
        args.val_csv_path,
        tuple(args.image_size),
        args.num_classes,
        transform
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = load_pretrained_mask2former(args.num_classes).to(device)
    
    # Start training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        num_epochs=args.epochs,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main()