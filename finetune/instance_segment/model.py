import io

import lightning as L
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import matplotlib

from finetune.instance_segment.CustomROIHeads import CustomMaskRCNN

matplotlib.use('Agg')
class MaskRCNNLightningModule(L.LightningModule):
    def __init__(self, num_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Load a pre-trained backbone
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        # Create the custom Mask R-CNN model
        self.model = CustomMaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            # Additional arguments if needed
        )

        # Replace the box predictor with a new one for your dataset
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor with a new one for your dataset
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=hidden_layer,
            num_classes=num_classes,
        )

        # Learning rate
        self.lr = lr

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0005,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        outputs = self.model(images)

        class_names = {1: "roof", 2: "building"}
        class_colors = {
            1: (0, 1, 0),  # Green for roof
            2: (0, 0, 1),  # Blue for building
        }

        if batch_idx == 0:
            num_images = min(5, len(images))
            for i in range(num_images):
                image = images[i]
                target = targets[i]
                output = outputs[i]

                masks_gt = target["masks"]
                labels_gt = target["labels"]
                boxes_gt = target["boxes"]
                offsets_gt_norm = target["offsets"]

                # Denormalize ground truth offsets
                widths_gt = boxes_gt[:, 2] - boxes_gt[:, 0]
                heights_gt = boxes_gt[:, 3] - boxes_gt[:, 1]
                widths_gt = torch.clamp(widths_gt, min=1e-6)
                heights_gt = torch.clamp(heights_gt, min=1e-6)
                offsets_gt = offsets_gt_norm.clone()
                offsets_gt[:, 0] = offsets_gt[:, 0] * widths_gt
                offsets_gt[:, 1] = offsets_gt[:, 1] * heights_gt

                masks_pred = output["masks"]
                labels_pred = output["labels"]
                boxes_pred = output["boxes"]
                scores_pred = output["scores"]
                offsets_pred = output["offsets"]

                score_threshold = 0.5
                keep = scores_pred >= score_threshold
                masks_pred = masks_pred[keep]
                labels_pred = labels_pred[keep]
                boxes_pred = boxes_pred[keep]
                offsets_pred = offsets_pred[keep]

                # Visualize Ground Truth
                fig_gt = visualize_masks(image, masks_gt, labels_gt, boxes_gt,
                                         offsets=offsets_gt,
                                         class_colors=class_colors,
                                         class_names=class_names,
                                         title="Ground Truth Masks and Boxes")

                # Visualize Predictions
                fig_pred = visualize_masks(image, masks_pred, labels_pred, boxes_pred,
                                           offsets_pred,
                                           class_colors, class_names, title="Predicted Masks and Boxes")

                # Log images to TensorBoard
                for fig, name in zip([fig_gt, fig_pred], ['gt', 'pred']):
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    img = transforms.ToTensor()(img)
                    self.logger.experiment.add_image(f"val/image_{i}_{name}", img, self.current_epoch)
                    plt.close(fig)

        return outputs


import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import matplotlib.patches as patches

def visualize_masks(image_tensor, masks, labels, boxes, offsets, class_colors, class_names, title):
    # Denormalize the image
    image = image_tensor.cpu()
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
        [0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()  # Convert to H x W x C numpy array

    # Initialize an empty mask
    H, W = image.shape[:2]
    combined_mask = np.zeros((H, W, 3), dtype=np.float32)

    alpha = 0.5

    # Create masks for each class
    for idx in range(len(labels)):
        mask = masks[idx].cpu().numpy()
        label = labels[idx].item()
        color = class_colors.get(label, (1, 1, 1))  # Default to white if label not found
        mask = np.squeeze(mask)
        mask_bool = mask > 0.5

        # Create a color mask
        color_mask = np.zeros_like(image)
        color_mask[:, :, 0] = color[0]
        color_mask[:, :, 1] = color[1]
        color_mask[:, :, 2] = color[2]

        # Apply alpha blending
        combined_mask[mask_bool] = combined_mask[mask_bool] * (1 - alpha) + color_mask[mask_bool] * alpha

    # Overlay masks onto the image
    image_with_masks = image * (1 - alpha) + combined_mask * alpha
    image_with_masks = np.clip(image_with_masks, 0, 1)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_with_masks)
    ax.set_title(title)
    ax.axis('off')

    # Draw bounding boxes and arrows
    for idx in range(len(boxes)):
        box = boxes[idx].cpu().numpy()
        label = labels[idx].item()
        color = class_colors.get(label, (1, 1, 1))

        # Create a Rectangle patch
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label text
        class_name = class_names.get(label, f"Class {label}")
        ax.text(x_min, y_min - 5, class_name, color=color, fontsize=6)

        # Draw the offset arrow if offsets are provided
        if offsets is not None:
            offset = offsets[idx].cpu().numpy()
            mask = masks[idx].cpu().numpy().squeeze()
            mask_indices = np.argwhere(mask > 0.5)
            if mask_indices.size > 0:
                y_center, x_center = mask_indices.mean(axis=0)
                # Adjust the scale if needed
                scale_factor = 1.0  # Adjust this value to scale the arrow length
                dx = offset[0] * scale_factor
                dy = offset[1] * scale_factor
                # Draw the arrow
                ax.quiver(x_center, y_center, -dx, -dy, angles='xy', scale_units='xy', scale=1, color='red', width=0.005)

    plt.tight_layout()

    return fig
