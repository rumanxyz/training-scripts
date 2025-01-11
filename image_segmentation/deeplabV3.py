"""
DeepLabV3 Semantic Segmentation Training Pipeline

This script provides functionality for training a DeepLabV3 model for semantic segmentation
tasks using polygon annotations. It supports multiple classes and handles data preparation,
training, and inference.

Required Data Format:
    - CSV files (train/validation) with columns:
        - image_path: Path to the image file
        - label: Path to the JSON annotation file

    - JSON annotation format:
    {
        "image_name": "image1.jpg",
        "polygons": [
            {"points": [[x1, y1], [x2, y2], ...], "class": 1},
            {"points": [[x3, y3], [x4, y4], ...], "class": 2}
        ]
    }
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
    Load a pre-trained DeepLabV3 model and adjust the classifier for the given number of classes.

    Args:
        num_classes: Number of classes, including background. For example, if the segmentation
                    involves "dog" and "cat", use 3 (0 for background, 1 for dog, 2 for cat).
        model_variant: The variant of DeepLabV3 model to load. Options: "mobilenet_v3_large" or "resnet101"

    Returns:
        Pretrained DeepLabV3 model adjusted for the specified number of classes.

    Raises:
        ValueError: If an unsupported model_variant is specified.
    """
    if model_variant == "mobilenet_v3_large":
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    elif model_variant == "resnet101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    else:
        raise ValueError(f"Unsupported model_variant: {model_variant}")

    # Adjust the classifier for the specified number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

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
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon['points'], dtype=np.int32)
        class_id = polygon['class']
        cv2.fillPoly(mask, [points], color=class_id)
    return mask

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
            annotation_data = json.load(f)
            polygons = annotation_data['polygons']

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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model_path = save_dir / "best_deeplabv3_lpt.pth"
    latest_model_path = save_dir / "latest_deeplabv3_lpt.pth"

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoints
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch+1}")

        torch.save(model.state_dict(), latest_model_path)

        if (epoch + 1) % save_interval == 0:
            save_path = save_dir / f"deeplabv3_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

def inference(
    image_path: str,
    model_path: str,
    image_size: Tuple[int, int],
    num_classes: int,
    model_variant: str = "mobilenet_v3_large"
) -> List[Dict]:
    """
    Perform inference on a single image and return polygon predictions.

    Args:
        image_path: Path to the input image
        model_path: Path to the trained model weights
        image_size: Tuple of (width, height) for input image resizing
        num_classes: Number of classes (including background)
        model_variant: DeepLabV3 model variant to use

    Returns:
        List of dictionaries containing predicted polygons and their class IDs
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_deeplabv3(num_classes, model_variant=model_variant)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
    prediction = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    polygons = []
    for class_id in range(1, num_classes):
        class_mask = (prediction == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            polygons.append({
                "class": class_id,
                "contour": cv2.approxPolyDP(cnt, epsilon=3, closed=True).squeeze().tolist()
            })

    return polygons

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
        num_epochs=args.epochs,
        save_interval=args.weight_save_interval
    )

if __name__ == "__main__":
    main()