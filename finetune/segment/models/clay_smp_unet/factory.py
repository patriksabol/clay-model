"""
Clay Segmentor for semantic segmentation tasks.

Attribution:
Decoder from Segformer: Simple and Efficient Design for Semantic Segmentation
with Transformers
Paper URL: https://arxiv.org/abs/2105.15203
"""

import re

import torch
from einops import rearrange, repeat
from torch import nn

from src.model import Encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import segmentation_models_pytorch as smp

class SegmentEncoder(Encoder):
    """
    Encoder class for segmentation tasks, incorporating a feature pyramid
    network (FPN).

    Attributes:
        feature_maps (list): Indices of layers to be used for generating
        feature maps.
        ckpt_path (str): Path to the clay checkpoint file.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        feature_maps,
        ckpt_path=None,
    ):
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )
        self.feature_maps = feature_maps

        # Define Feature Pyramid Network (FPN) layers
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fpn5 = nn.Identity()

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Load model from checkpoint if provided
        self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path):
        """
        Load the model's state from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }

            # Load the modified state dict into the model
            model_state_dict = self.state_dict()
            for name, param in new_state_dict.items():
                if (
                    name in model_state_dict
                    and param.size() == model_state_dict[name].size()
                ):
                    model_state_dict[name].copy_(param)
                else:
                    print(f"No matching parameter for {name} with size {param.size()}")

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False

    def forward(self, datacube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B, C, H, W = cube.shape

        # Patchify and create embeddings per patch
        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        features = []
        i = 0
        for idx, (attn, ff) in enumerate(self.transformer.layers):
            patches = attn(patches) + patches
            patches = ff(patches) + patches
            if idx in self.feature_maps:
                _cube = rearrange(
                    patches[:, 1:, :], "B (H W) D -> B D H W", H=H // 8, W=W // 8
                )
                features.append(_cube)
                i = i + 1
        patches = self.transformer.norm(patches)
        _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=H // 8, W=W // 8)
        features.append(_cube)

        return features


class Segmentor(nn.Module):
    """
    Clay Segmentor class that combines the Encoder with FPN layers for semantic
    segmentation.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
        feature_maps (list): Indices of layers to be used for generating feature maps.
        ckpt_path (str): Path to the checkpoint file.
    """

    def __init__(self, num_classes, feature_maps, ckpt_path):
        super().__init__()
        # Default values are for the clay mae base model.
        dim = 768
        self.encoder = SegmentEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=dim,
            depth=12,
            heads=12,
            dim_head=64,
            mlp_ratio=4.0,
            feature_maps=feature_maps,
            ckpt_path=ckpt_path,
        )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn4 = nn.Identity()

        self.fpn5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder_channels = [256, 128, 64, 32]
        self._out_channels = [self.encoder.dim] * 5
        self.decoder = UnetDecoder(
            encoder_channels=self._out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=len(self.decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, datacube):
        """
        Forward pass of the Segmentor.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        features = self.encoder(datacube)
        features = features[::-1]
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4, self.fpn5]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        decoder_output = self.decoder(*features)
        logits = self.segmentation_head(decoder_output)
        return logits
