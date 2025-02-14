"""
DeepLabV3 Semantic Segmentation Training Pipeline

This script provides functionality for training a DeepLabV3 model for semantic segmentation
tasks using polygon annotations. It supports multiple classes and handles data preparation,
training, and inference.

Required Data Format:
    - CSV files (train/validation) with columns:
        - image_path: Path to the image file
        - label: Path to the JSON annotation file

    - JSON file annotation format:
    [
        {"points": [[x1, y1], [x2, y2], ...], "class": 1},
        {"points": [[x3, y3], [x4, y4], ...], "class": 2}
    ]
    Note: Class IDs should be from 1 to num_classes. Class 0 is reserved for background.
"""

import cv2
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

def load_pretrained_deeplabv3(num_classes: int, model_variant: str = "mobilenet_v3_large") -> nn.Module:
    """
    Load a pre-trained DeepLabV3 model with proper initialization for semantic segmentation.
    
    Args:
        num_classes: For binary segmentation (one object), use 2 (background=0, object=1)
                    For multi-class, use (N+1) where N is number of objects
        model_variant: Model backbone variant
    
    Returns:
        Properly initialized DeepLabV3 model
    """
    # Load with weights=default instead of pretrained=True for newer torchvision
    if model_variant == "mobilenet_v3_large":
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="COCO_WITH_VOC_LABELS_V1")
    elif model_variant == "resnet101":
        model = models.segmentation.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
    else:
        raise ValueError(f"Unsupported model_variant: {model_variant}")
    
    # Get number of input channels in classifier's last layer
    in_channels = model.classifier[-1].in_channels
    
    # Replace the classifier's last layer with proper initialization
    model.classifier[-1] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=num_classes,
        kernel_size=1,
        bias=True
    )
    
    # Initialize the new layer with better weights
    nn.init.xavier_uniform_(model.classifier[-1].weight)
    if model.classifier[-1].bias is not None:
        nn.init.constant_(model.classifier[-1].bias, 0)
        
    return model

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # predictions shape: (B, H, W)
        # targets shape: (B, H, W)
        
        # Flatten the tensors while keeping batch dimension
        predictions = predictions.reshape(predictions.shape[0], -1)  # (B, H*W)
        targets = targets.reshape(targets.shape[0], -1)  # (B, H*W)
        
        # Compute Dice score for each sample in batch
        intersection = (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean Dice loss across batch
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Combines Cross Entropy and Dice Loss for better segmentation results.
    """
    def __init__(self, num_classes, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        
    def forward(self, outputs, targets):
        # outputs shape: (B, C, H, W)
        # targets shape: (B, H, W)
        
        # Cross Entropy Loss
        ce_loss = self.ce_loss(outputs, targets)
        
        # Dice Loss - compute for each class
        predictions = torch.softmax(outputs, dim=1)  # (B, C, H, W)
        dice_loss = 0
        
        # Skip background class (index 0)
        for cls in range(1, self.num_classes):
            class_pred = predictions[:, cls]  # (B, H, W)
            class_target = (targets == cls).float()  # (B, H, W)
            dice_loss += self.dice_loss(class_pred, class_target)
        
        # Average Dice loss over classes
        dice_loss = dice_loss / (self.num_classes - 1)
        
        # Combine losses
        return ce_loss * (1 - self.dice_weight) + dice_loss * self.dice_weight


def create_segmentation_mask(image_shape: Tuple[int, int], polygons: List[Dict]) -> np.ndarray:
    """
    Convert polygons with class labels to a multi-class segmentation mask.

    Args:
        image_shape: Tuple of (height, width) for the output mask
        polygons: List of dictionaries containing polygon points and class IDs
                 Format: [{"points": [[x1, y1], [x2, y2], ...], "class": class_id}, ...]

    Returns:
        numpy.ndarray: Segmentation mask where pixel values represent class IDs (0 for background)
    """
    imgh, imgw = image_shape
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.array([[int(cord[0]*imgw), int(cord[1]*imgh)] for cord in polygon["points"]], dtype=np.int32)
        class_id = polygon['class']
        cv2.fillPoly(mask, [points], color=class_id)
    return mask


# def calculate_metrics(predictions, targets):
#     predictions = predictions.argmax(1)
#     correct = (predictions == targets).float()
#     accuracy = correct.mean()
    
#     # Calculate per-class IoU
#     ious = []
#     for cls in range(predictions.shape[1]):
#         pred_cls = predictions == cls
#         target_cls = targets == cls
#         intersection = (pred_cls & target_cls).float().sum()
#         union = (pred_cls | target_cls).float().sum()
#         iou = (intersection + 1e-6) / (union + 1e-6)
#         ious.append(iou)
    # return accuracy, torch.tensor(ious).mean()


def calculate_metrics(predictions, targets, num_classes, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):
    """
    Calculate class-wise accuracy and percentage of TPs at different IoU thresholds.

    Args:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth masks (B, H, W)
        thresholds: List of IoU thresholds to evaluate

    Returns:
        metrics: Dictionary containing class-wise accuracy and percentage of TPs at thresholds
    """
    predictions = predictions.argmax(1)  # (B, H, W)
    num_samples = predictions.shape[0]  # Number of samples in the validation set

    # Initialize metrics dictionary
    metrics = {
        "accuracy": 0,
        "class_metrics": {
            cls: {f"IoU@{int(t*100)}%": 0 for t in thresholds} for cls in range(num_classes)
        }
    }

    # Calculate overall accuracy
    correct = (predictions == targets).float()
    metrics["accuracy"] = correct.mean().item()

    # Calculate class-wise percentage of TPs at thresholds
    for cls in range(num_classes):
        pred_cls = predictions == cls
        target_cls = targets == cls

        # Calculate IoU for each sample in the batch
        for sample_idx in range(num_samples):
            intersection = (pred_cls[sample_idx] & target_cls[sample_idx]).float().sum()
            union = (pred_cls[sample_idx] | target_cls[sample_idx]).float().sum()
            iou = (intersection + 1e-6) / (union + 1e-6)

            # Check IoU against thresholds for this sample
            for threshold in thresholds:
                if iou >= threshold:
                    metrics["class_metrics"][cls][f"IoU@{int(threshold*100)}%"] += 1

        # Normalize TP counts by the total number of samples to get percentages
        for threshold in thresholds:
            metrics["class_metrics"][cls][f"IoU@{int(threshold*100)}%"] /= num_samples

    return metrics

class CustomSegmentationDataset(Dataset):
    """
    Custom dataset for semantic segmentation with polygon annotations.

    Args:
        csv_file_path: Path to CSV file containing image paths and annotation paths
        image_size: Tuple of (width, height) for resizing images
        transform: Optional transforms to be applied to images

    The CSV file should have two columns:
        - image_path: Path to the image file
        - label: Path to the JSON annotation file
    """

    def __init__(self, csv_file_path: str, image_size: Tuple[int, int], transform: Optional[transforms.Compose] = None):
        self.df_data = pd.read_csv(csv_file_path)
        self.image_files = self.df_data.image_path.values
        self.image_annotations = self.df_data.label.values
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size)

        # Load annotations
        label_path = self.image_annotations[idx]
        with open(label_path, 'r') as f:
            polygons = json.load(f)

        # Create segmentation mask
        mask = create_segmentation_mask(image.size[::-1], polygons)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

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
    """
    Train the DeepLabV3 model and save checkpoints.

    Args:
        model: DeepLabV3 model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        save_dir: Directory to save model checkpoints
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        save_interval: Interval (in epochs) to save model checkpoints
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()

    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Use combined loss for better handling of class imbalance
    criterion = CombinedLoss(num_classes=model.classifier[-1].out_channels)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float('inf')
    best_model_path = save_dir / "best_deeplabv3_lpt.pth"
    latest_model_path = save_dir / "latest_deeplabv3_lpt.pth"

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metrics = {'loss': 0, 'accuracy': 0, 'iou': 0}
        running_loss = 0.0
        # for images, masks in train_loader:
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']

            # Check prediction distribution periodically
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    pred_classes = outputs.argmax(1)
                    unique_classes = torch.unique(pred_classes)
                    print(f"\nTrain Batch {batch_idx} predictions - Unique classes: {unique_classes}")
                    print(f"Train Batch Masks - Unique classes: {torch.unique(masks)}")

            loss = criterion(outputs, masks)


            running_loss += loss.item()

            # Calculate metrics
            # metrics = calculate_metrics(outputs, masks, num_classes)
            
            # print(f"Train Batch {batch_idx} :\nBatch Loss :{loss.item()} \nClass Wise IoU Metrics :")
            # for cls, cls_metrics in metrics["class_metrics"].items():
            #     print(f"Class {cls}: {cls_metrics}")

            train_metrics['loss'] += loss.item()
            # train_metrics['accuracy'] += metrics['accuracy']

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        train_loss = running_loss / len(train_loader)
        # Normalize metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Accumulate predictions and targets
                all_preds.append(outputs)
                all_targets.append(masks)

                # Monitor predictions
                pred_classes = outputs.argmax(1)
                print(f"Val Batch - Unique predicted classes: {torch.unique(pred_classes)}")
                print(f"Val Batch - Unique mask classes: {torch.unique(masks)}")

        val_loss = val_loss / len(val_loader)

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metrics on full validation set
        val_metrics = calculate_metrics(all_preds, all_targets, num_classes)

        # Print metrics
        print("Train and Validation Summary :")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_loss:.4f}")
        # print(f"Train Accuracy: {train_metrics['accuracy']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        print("Val Class-Wise Metrics:")
        for cls, cls_metrics in val_metrics["class_metrics"].items():
            print(f"Class {cls}: {cls_metrics}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'num_classes': model.classifier[-1].out_channels
            }, best_model_path)
            print(f"Saved best model at epoch {epoch+1}")

        # save every epoch checkpoint
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'num_classes': model.classifier[-1].out_channels
            }, latest_model_path)
        print(f"Saved latest model at epoch {epoch+1}")

        if (epoch + 1) % save_interval == 0:
            save_path = save_dir / f"deeplabv3_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'num_classes': model.classifier[-1].out_channels
            }, save_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 model for semantic segmentation")
    parser.add_argument('--train_csv_path', type=str, required=True,
                      help="Path to training CSV file with image_path and label columns")
    parser.add_argument('--val_csv_path', type=str, required=True,
                      help="Path to validation CSV file with image_path and label columns")
    parser.add_argument('--num_classes', type=int, required=True,
                      help="Number of classes including background (num_object_classes + 1)")
    parser.add_argument('--model_variant', type=str, default="mobilenet_v3_large",
                      choices=["mobilenet_v3_large", "resnet101"])
    parser.add_argument('--image_size', type=int, nargs=2, default=[480, 480],
                      help='Image size as width height')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--chckp_save_directory', type=str, default="chpk/")
    parser.add_argument('--weight_save_interval', type=int, default=5,
                      help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Validate num_classes
    if args.num_classes < 2:
        raise ValueError("num_classes must be at least 2 (1 for background, 1+ for objects)")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and data loaders
    train_dataset = CustomSegmentationDataset(args.train_csv_path, tuple(args.image_size), transform)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    val_dataset = CustomSegmentationDataset(args.val_csv_path, tuple(args.image_size), transform)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = load_pretrained_deeplabv3(args.num_classes, model_variant=args.model_variant)
    model = model.to(device)

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.chckp_save_directory,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        num_epochs=args.epochs,
        save_interval=args.weight_save_interval
    )

if __name__ == "__main__":
    main()