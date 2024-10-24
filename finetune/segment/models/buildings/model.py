"""
LightningModule for training and validating a segmentation model using the
Segmentor class.
"""

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
import io
from PIL import Image
import torchvision.transforms as transforms

class LightingSegmentor(L.LightningModule):
    """
    LightningModule for segmentation tasks, utilizing Clay Segmentor.

    Attributes:
        segmentation_model (nn.Module): Clay Segmentor model.
        loss_fn (nn.Module): The loss function.
        iou (Metric): Intersection over Union metric.
        f1 (Metric): F1 Score metric.
        lr (float): Learning rate.
    """

    def __init__(  # # noqa: PLR0913
        self, model_arch, encoder_name, encoder_weights, in_channels, lr, wd, b1, b2
    ):
        super().__init__()
        classes = 2
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        # Create the segmentation model using SMP
        self.segmentation_model = smp.create_model(
            arch=model_arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            in_channels=in_channels,
            activation=None,  # No activation since we're using logits
        )
        out_channels = self.get_out_channels(self.segmentation_model.decoder)

        # Segmentation head
        self.segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=out_channels,
            out_channels=2,  # Rooftop and building masks
            activation=None,
            kernel_size=3,
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Conv2d(out_channels + 2,  # because concatenate segmentation features with shift vectors
                      out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 2, kernel_size=3, padding=1),  # Output x_shift and y_shift
            nn.Tanh()
        )

        # Loss functions
        self.segmentation_loss_fn = nn.BCEWithLogitsLoss()
        self.regression_loss_fn = nn.SmoothL1Loss(reduction='none')  # We'll handle masking manually

        # Metrics
        self.iou = BinaryJaccardIndex()
        self.f1 = BinaryF1Score()

    @staticmethod
    def get_out_channels(module):
        """Method reused from
        https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/blob/master/frame_field_learning/model.py

        Args:
            module ([type]): [description]

        Returns:
            [type]: [description]
        """
        if hasattr(module, "out_channels"):
            return module.out_channels
        children = list(module.children())
        i = 1
        out_channels = None
        while out_channels is None and i <= len(children):
            last_child = children[-i]
            out_channels = LightingSegmentor.get_out_channels(last_child)
            i += 1
        # If we get out of the loop but out_channels is None,
        # then the prev child of the parent module will be checked, etc.
        return out_channels

    def forward(self, sample):
        x = sample["orto"]
        features = self.segmentation_model.encoder(x)
        decoder_output = self.segmentation_model.decoder(*features)

        # segmentation stuff
        segmentation_output = self.segmentation_model.segmentation_head(decoder_output)
        # regression stuff
        segmentation_features = segmentation_output.clone().detach()
        regression_input = torch.cat([decoder_output, segmentation_features], dim=1)
        regression_output = self.regression_head(regression_input)

        return segmentation_output, regression_output


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.segmentation_model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=1000,
            T_mult=1,
            eta_min=self.hparams.lr * 100,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        # Get model outputs
        segmentation_output, regression_output = self(batch)

        # Get targets from the batch
        rooftop_mask = batch["label_roof"]  # Shape: (B, H, W)
        building_mask = batch["label_building"]  # Shape: (B, H, W)
        shift_vectors = batch["label_shift"]  # Shape: (B, 2, H, W)

        # Ensure the targets are on the correct device
        rooftop_mask = rooftop_mask.to(self.device)
        building_mask = building_mask.to(self.device)
        shift_vectors = shift_vectors.to(self.device)

        # Prepare segmentation targets by stacking the masks
        segmentation_targets = torch.stack([rooftop_mask, building_mask], dim=1).float()  # Shape: (B, 2, H, W)

        # Compute segmentation loss
        seg_loss = self.segmentation_loss_fn(segmentation_output, segmentation_targets)

        # Apply sigmoid to get probabilities
        seg_output_probs = torch.sigmoid(segmentation_output)
        preds_rooftop_probs = seg_output_probs[:, 0, :, :]  # Assuming rooftop is at index 0

        # Binarize outputs for masking (threshold can be adjusted)
        preds_rooftop_mask = (preds_rooftop_probs > 0.5).unsqueeze(1).float()  # Shape: (B, 1, H, W)

        # Compute regression loss only at predicted rooftop pixels
        regression_loss = self.regression_loss_fn(regression_output, shift_vectors)
        regression_loss = regression_loss * preds_rooftop_mask  # Mask the loss with predicted mask

        # Normalize the loss by the number of predicted rooftop pixels
        num_pred_rooftop_pixels = preds_rooftop_mask.sum() + 1e-6  # Avoid division by zero
        regression_loss = regression_loss.sum() / num_pred_rooftop_pixels

        # regularization loss
        non_rooftop_mask = 1.0 - preds_rooftop_mask
        regression_output_magnitude = torch.norm(regression_output, dim=1, keepdim=True)
        regularization_loss = (regression_output_magnitude * non_rooftop_mask).sum() / (non_rooftop_mask.sum() + 1e-6)

        # Total loss (you can adjust the weighting if needed)
        total_loss = seg_loss + regression_loss + 0.1 * regularization_loss

        # Compute metrics for segmentation
        # Apply sigmoid to get probabilities
        seg_output_probs = torch.sigmoid(segmentation_output)

        # Binarize outputs
        preds_rooftop = (seg_output_probs[:, 0, :, :] > 0.5).int()
        preds_building = (seg_output_probs[:, 1, :, :] > 0.5).int()

        targets_rooftop = rooftop_mask.int()
        targets_building = building_mask.int()

        # Compute IoU and F1 for rooftop mask
        iou_rooftop = self.iou(preds_rooftop, targets_rooftop)
        f1_rooftop = self.f1(preds_rooftop, targets_rooftop)

        # Compute IoU and F1 for building mask
        iou_building = self.iou(preds_building, targets_building)
        f1_building = self.f1(preds_building, targets_building)
        avg_iou = (iou_rooftop + iou_building) / 2

        # Log losses
        self.log(f"{phase}/seg_loss", seg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{phase}/regression_loss", regression_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log(f"{phase}/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)

        # Log metrics
        self.log(f"{phase}/iou_rooftop", iou_rooftop, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log(f"{phase}/f1_rooftop", f1_rooftop, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log(f"{phase}/iou_building", iou_building, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log(f"{phase}/f1_building", f1_building, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log(f"{phase}/iou", avg_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Log images to TensorBoard
        if phase == "val" and batch_idx == 0:  # Log only for the first batch of each epoch in the validation phase
            num_images = min(5, batch["orto"].size(0))

            for i in range(num_images):
                # Denormalize image for visualization
                original_image = batch["orto"][i].to("cpu")
                original_image = original_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                    [0.485, 0.456, 0.406]).view(3, 1, 1)

                # Get masks and predictions
                rooftop_mask = batch["label_roof"][i].to("cpu")
                building_mask = batch["label_building"][i].to("cpu")
                preds_rooftop = (torch.sigmoid(segmentation_output[:, 0, :, :]) > 0.5).int().to("cpu")[i]
                preds_building = (torch.sigmoid(segmentation_output[:, 1, :, :]) > 0.5).int().to("cpu")[i]
                regression_output_img = regression_output[i].detach().cpu().numpy()

                # Get shift vectors (ground truth for regression)
                shift_vector_gt = shift_vectors[i].to("cpu").numpy()

                # Plot the original image
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes[0, 0].imshow(np.transpose(original_image.numpy(), (1, 2, 0)))
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis('off')

                # Plot ground truth rooftop mask
                axes[0, 1].imshow(rooftop_mask.numpy(), cmap='gray')
                axes[0, 1].set_title("Ground Truth Rooftop Mask")
                axes[0, 1].axis('off')

                # Plot predicted rooftop mask
                axes[0, 2].imshow(preds_rooftop.numpy(), cmap='gray')
                axes[0, 2].set_title("Predicted Rooftop Mask")
                axes[0, 2].axis('off')

                # Plot ground truth building mask
                axes[0, 3].imshow(building_mask.numpy(), cmap='gray')
                axes[0, 3].set_title("Ground Truth Building Mask")
                axes[0, 3].axis('off')

                # Plot predicted building mask
                axes[1, 1].imshow(preds_building.numpy(), cmap='gray')
                axes[1, 1].set_title("Predicted Building Mask")
                axes[1, 1].axis('off')

                # Plot ground truth shift vector magnitude (for regression)
                shift_gt_magnitude = np.linalg.norm(shift_vector_gt, axis=0)
                im1 = axes[1, 2].imshow(shift_gt_magnitude, cmap='viridis')
                axes[1, 2].set_title("Ground Truth Shift Vectors")
                axes[1, 2].axis('off')
                fig.colorbar(im1, ax=axes[1, 2], fraction=0.046,
                             pad=0.04)  # Adding colorbar for ground truth shift vector

                # Plot predicted shift vector magnitude (regression output)
                predicted_shift_magnitude = np.linalg.norm(regression_output_img, axis=0)
                im2 = axes[1, 3].imshow(predicted_shift_magnitude, cmap='viridis')
                axes[1, 3].set_title("Predicted Shift Vectors")
                axes[1, 3].axis('off')
                fig.colorbar(im2, ax=axes[1, 3], fraction=0.046, pad=0.04)  # Adding colorbar for predicted shift vector

                # Remove any unused subplot (optional)
                axes[1, 0].axis('off')
                fig.tight_layout()

                # Save the plot to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)

                # Convert buffer to a PIL image
                image = Image.open(buf)

                # Convert PIL image to tensor
                image_tensor = transforms.ToTensor()(image)

                # Log the image to TensorBoard
                self.logger.experiment.add_image(f"{phase}/segmentation_regression_output_{i}", image_tensor,
                                                 global_step=self.current_epoch)

                # Close the buffer
                buf.close()

        return total_loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")
