"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from skimage.measure import label
import torch
from torchvision.ops import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import json
import cv2

# Suppress NotGeoreferencedWarning from rasterio
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class BuildingDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        images_paths (list): List of paths to the image files.
        labels_paths (list): List of paths to the label files.
    """

    def __init__(self, images_paths, labels_paths):
        self.images_paths = images_paths
        self.labels_paths = labels_paths

        self.transform = self.create_transforms(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
            ],
        )

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images_paths[idx])
        image = np.array(image).astype(np.float32) / 255.0  # H x W x C

        # Convert image to tensor and normalize
        img = torch.tensor(image).permute(2, 0, 1)  # C x H x W
        img = self.transform(img)
        img = tv_tensors.Image(img)

        # Load JSON label
        with open(self.labels_paths[idx], 'r') as f:
            annotation = json.load(f)

        masks = []
        boxes = []
        labels = []

        for annot in annotation['annotations']:
            if annot['ignore'] == 0.0:
                roof = annot['roof']
                footprint = annot['footprint']

                # Create mask for the roof
                mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
                roof_polygon = np.array(roof).reshape(-1, 2)
                cv2.fillPoly(mask, [roof_polygon], 1)
                masks.append(mask)

                # Calculate bounding box combining footprint and roof
                combined_polygon = np.vstack((roof_polygon, np.array(footprint).reshape(-1, 2)))
                x_min, y_min = np.min(combined_polygon, axis=0)
                x_max, y_max = np.max(combined_polygon, axis=0)
                boxes.append([x_min, y_min, x_max, y_max])

                labels.append(1)  # Assuming 1 is the label for buildings

        masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Prepare target in the required format
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        return img, target


class BuildingDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        img_dir (str): Directory containing image chips.
        label_dir (str): Directory containing labels.
        split_file_path (str): Path to the split file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
    """

    def __init__(
        self,
        img_dir,
        label_dir,
        split_file_path,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.split_file_path = split_file_path
        self.splits = self.parse_split_file(split_file_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        train_images, train_labels = self.filter_paths(self.splits['train'])
        val_images, val_labels = self.filter_paths(self.splits['valid'])

        if stage in {"fit", None}:
            self.trn_ds = BuildingDataset(train_images, train_labels)
            self.val_ds = BuildingDataset(val_images, val_labels)

        if stage == "validate":
            self.val_ds = BuildingDataset(val_images, val_labels)

    def filter_paths(self, split_filenames):
        """
        Filters images and labels based on filenames from the split file.

        Args:
            split_filenames (list): List of filenames from the split file.

        Returns:
            tuple: Lists of image and label paths that match the filenames.
        """
        images = []
        labels = []
        for filename in split_filenames:
            image_path = self.img_dir / f"{filename}.png"
            label_path = self.label_dir / f"{filename}.json"
            if image_path.exists() and label_path.exists():
                images.append(image_path)
                labels.append(label_path)
        return images, labels

    @staticmethod
    def parse_split_file(split_file_path):
        split = {'train': [], 'valid': []}
        with open(split_file_path, 'r') as f:
            for line in f:
                subset, filename = line.strip().split(':')
                # remove .png extension
                filename = re.sub(r'\.png$', '', filename)
                split[subset].append(filename)
        return split

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

def collate_fn(batch):
    return tuple(zip(*batch))
