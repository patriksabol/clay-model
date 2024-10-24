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

class BuildingDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, orto_dir, label_roof_dir, label_building_dir, label_shift_dir):
        self.orto_dir = Path(orto_dir)
        self.label_roof_dir = Path(label_roof_dir)
        self.label_building_dir = Path(label_building_dir)
        self.label_shift_dir = Path(label_shift_dir)

        # Load chip and label file names
        self.ortos = [orto_path.name for orto_path in self.orto_dir.glob("*.tif")]
        self.label_roofs = [label_roof_path.name for label_roof_path in self.label_roof_dir.glob("*.tif")]
        self.label_buildings = [label_building_path.name for label_building_path in self.label_building_dir.glob("*.tif")]
        self.label_shifts = [label_shift_path.name for label_shift_path in self.label_shift_dir.glob("*.tif")]

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
        assert len(self.ortos) == len(self.label_roofs) == len(self.label_buildings) == len(self.label_shifts), f"Number of chips and labels do not match. len(ortos): {len(self.ortos)}, len(label_roofs): {len(self.label_roofs)}, len(label_buildings): {len(self.label_buildings)}, len(label_shifts): {len(self.label_shifts)}"
        return len(self.ortos)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        # use PIL to load the image
        orto = Image.open(self.orto_dir / self.ortos[idx])
        orto = np.array(orto).astype(np.float32) / 255.0
        label_roof = np.array(Image.open(self.label_roof_dir / self.label_roofs[idx]).convert("L"))
        label_roof = (label_roof > 0).astype(np.float32)
        label_building = np.array(Image.open(self.label_building_dir / self.label_buildings[idx]).convert("L"))
        label_building = (label_building > 0).astype(np.float32)
        # label_shift = np.array(Image.open(self.label_shift_dir / self.label_shifts[idx]).convert("L"))
        # open label_shift using rasterio
        label_shift = rasterio.open(self.label_shift_dir / self.label_shifts[idx]).read([1, 2])
        label_shift = label_shift.astype(np.float32) / label_shift.shape[1]

        sample = {
            "orto": self.transform(torch.tensor(orto).permute(2, 0, 1)),
            "label_roof": torch.tensor(label_roof),
            "label_building": torch.tensor(label_building),
            "label_shift": torch.tensor(label_shift),
        }

        return sample


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
        self.train_label_shift_dir = train_root_dir / "rooftop_shift" / "images"

        self.val_chip_dir = val_root_dir / "orto" / "images"
        self.val_label_roof_dir = val_root_dir / "rooftop_mask" / "images"
        self.val_label_building_dir = val_root_dir / "whole_buildings" / "images"
        self.val_label_shift_dir = val_root_dir / "rooftop_shift" / "images"

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
                self.train_label_shift_dir,
            )
            self.val_ds = BuildingDataset(
                self.val_chip_dir,
                self.val_label_roof_dir,
                self.val_label_building_dir,
                self.val_label_shift_dir,
            )
        if stage == "validate":
            self.val_ds = BuildingDataset(
                self.val_chip_dir,
                self.val_label_roof_dir,
                self.val_label_building_dir,
                self.val_label_shift_dir,
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
        )
