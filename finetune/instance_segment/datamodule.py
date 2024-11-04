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

# Suppress NotGeoreferencedWarning from rasterio
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class BuildingDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, orto_dir, label_roof_dir, label_building_dir):
        self.orto_dir = Path(orto_dir)
        self.label_roof_dir = Path(label_roof_dir)
        self.label_building_dir = Path(label_building_dir)

        # Load chip and label file names
        self.ortos = [orto_path.name for orto_path in self.orto_dir.glob("*.tif")]
        self.label_roofs = [label_roof_path.name for label_roof_path in self.label_roof_dir.glob("*.tif")]
        self.label_buildings = [label_building_path.name for label_building_path in self.label_building_dir.glob("*.tif")]

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
        assert len(self.ortos) == len(self.label_roofs) == len(self.label_buildings), f"Number of chips and labels do not match. len(ortos): {len(self.ortos)}, len(label_roofs): {len(self.label_roofs)}, len(label_buildings): {len(self.label_buildings)}"
        return len(self.ortos)

    def __getitem__(self, idx):
        # Load image
        orto = Image.open(self.orto_dir / self.ortos[idx])
        orto = np.array(orto).astype(np.float32) / 255.0  # H x W x C

        # Convert image to tensor and normalize
        img = torch.tensor(orto).permute(2, 0, 1)  # C x H x W
        img = self.transform(img)

        # Load roof and building masks
        label_roof = np.array(
            Image.open(self.label_roof_dir / self.label_roofs[idx]).convert("L")
        )
        label_roof = (label_roof > 0).astype(np.uint8)

        label_building = np.array(
            Image.open(self.label_building_dir / self.label_buildings[idx]).convert("L")
        )
        label_building = (label_building > 0).astype(np.uint8)

        masks = []
        labels = []

        # Process roof instances
        roof_instances = label(label_roof)
        num_roof_instances = roof_instances.max()
        for i in range(1, num_roof_instances + 1):
            mask = (roof_instances == i).astype(np.uint8)
            masks.append(mask)
            labels.append(1)  # Label for roof

        # Process building instances
        building_instances = label(label_building)
        num_building_instances = building_instances.max()
        for i in range(1, num_building_instances + 1):
            mask = (building_instances == i).astype(np.uint8)
            masks.append(mask)
            labels.append(2)  # Label for building

        # Convert masks and labels to tensors
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)  # [num_objs, H, W]
        labels = torch.tensor(np.array(labels), dtype=torch.int64)

        # Get bounding boxes for each mask
        boxes = masks_to_boxes(masks)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Wrap image into torchvision tv_tensor
        img = tv_tensors.Image(img)

        # Prepare target dictionary
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        return img, target


class BuildingDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_root_dir,
        val_root_dir,
        batch_size,
        num_workers,
    ):
        super().__init__()
        train_root_dir = Path(train_root_dir)
        val_root_dir = Path(val_root_dir)
        self.train_chip_dir = train_root_dir / "orto" / "images"
        self.train_label_roof_dir = train_root_dir / "rooftop_mask" / "images"
        self.train_label_building_dir = train_root_dir / "whole_buildings" / "images"

        self.val_chip_dir = val_root_dir / "orto" / "images"
        self.val_label_roof_dir = val_root_dir / "rooftop_mask" / "images"
        self.val_label_building_dir = val_root_dir / "whole_buildings" / "images"

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = BuildingDataset(
                self.train_chip_dir,
                self.train_label_roof_dir,
                self.train_label_building_dir,
            )
            self.val_ds = BuildingDataset(
                self.val_chip_dir,
                self.val_label_roof_dir,
                self.val_label_building_dir,
            )
        if stage == "validate":
            self.val_ds = BuildingDataset(
                self.val_chip_dir,
                self.val_label_roof_dir,
                self.val_label_building_dir,
            )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

def collate_fn(batch):
    return tuple(zip(*batch))