import torch
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference
from torchvision.ops import MultiScaleRoIAlign

class CustomMaskRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes, **kwargs):
        super().__init__(backbone, num_classes, **kwargs)

        # Access the existing roi_heads to extract parameters
        existing_roi_heads = self.roi_heads

        # Replace the RoI heads with your custom heads
        self.roi_heads = CustomRoIHeads(
            # Box head components
            box_roi_pool=existing_roi_heads.box_roi_pool,
            box_head=existing_roi_heads.box_head,
            box_predictor=existing_roi_heads.box_predictor,
            # Mask head components
            mask_roi_pool=existing_roi_heads.mask_roi_pool,
            mask_head=existing_roi_heads.mask_head,
            mask_predictor=existing_roi_heads.mask_predictor,
            # Box training parameters
            fg_iou_thresh=existing_roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=existing_roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=existing_roi_heads.box_coder.weights,
            # Box inference parameters
            score_thresh=existing_roi_heads.score_thresh,
            nms_thresh=existing_roi_heads.nms_thresh,
            detections_per_img=existing_roi_heads.detections_per_img,
            # Offset head components
            offset_roi_pool=offset_roi_pool,
            offset_head=OffsetHead(in_channels=256),
            offset_predictor=OffsetPredictor(in_channels=256 * 7 * 7),
            # Additional arguments if any
            **kwargs,
        )


# Offset ROI Pooler
offset_roi_pool = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2
)

# Offset Head (a simple CNN)
class OffsetHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

# Offset Predictor (fully connected layer)
class OffsetPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, 2)  # Adjust according to your ROI size

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class CustomRoIHeads(RoIHeads):
    def __init__(
        self,
        # Box head components
        box_roi_pool,
        box_head,
        box_predictor,
        # Mask head components (optional)
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        # Offset head components
        offset_roi_pool=None,
        offset_head=None,
        offset_predictor=None,
        # Box training parameters
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        # Box inference parameters
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        # Additional keyword arguments
        **kwargs,
    ):
        super().__init__(
            # Box head components
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            # Box training parameters
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            # Box inference parameters
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            # Mask head components
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
            **kwargs,
        )

        # Initialize the offset head components
        self.offset_roi_pool = offset_roi_pool
        self.offset_head = offset_head
        self.offset_predictor = offset_predictor

    def forward(self,
                features,
                proposals,
                image_shapes,
                targets=None):
        """
        Arguments:
            features (Dict[str, Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])

        Returns:
            result (List[Dict])
            losses (Dict[str, Tensor])
        """
        # Original code from torchvision's RoIHeads.forward
        # Modified to include the offset head

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")

            # Match the proposals with the ground truth
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # Compute box head outputs
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}

        if self.training:
            # Compute losses for the box head
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses.update({"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg})
        else:
            # Postprocess the detections
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            result = []
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
            # For mask and offset heads, we need the boxes
            proposals = [r["boxes"] for r in result]

        # Mask head
        if self.has_mask():
            mask_losses = {}
            if self.training:
                assert targets is not None

                # Use positive proposals for mask head
                num_images = len(proposals)
                all_num_pos = 0
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos_inds = torch.where(labels[img_id] > 0)[0]
                    all_num_pos += pos_inds.numel()
                    mask_proposals.append(proposals[img_id][pos_inds])
                    pos_matched_idxs.append(matched_idxs[img_id][pos_inds])

                if all_num_pos > 0:
                    # Extract mask features
                    mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                    mask_features = self.mask_head(mask_features)
                    mask_logits = self.mask_predictor(mask_features)

                    # Compute mask loss
                    gt_masks = [t["masks"] for t in targets]
                    gt_labels = [t["labels"] for t in targets]
                    rcnn_loss_mask = maskrcnn_loss(
                        mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                    mask_losses = {"loss_mask": rcnn_loss_mask}
                else:
                    mask_losses = {"loss_mask": torch.tensor(0.0, device=features[0].device)}
            else:
                # Inference
                if len(proposals) > 0:
                    mask_features = self.mask_roi_pool(features, proposals, image_shapes)
                    mask_features = self.mask_head(mask_features)
                    mask_logits = self.mask_predictor(mask_features)

                    labels = [r["labels"] for r in result]
                    masks_probs = maskrcnn_inference(mask_logits, labels)
                    for mask_prob, r in zip(masks_probs, result):
                        r["masks"] = mask_prob
                else:
                    for r in result:
                        r["masks"] = torch.empty((0, 1, *image_shapes[0]), device=proposals[0].device)
            losses.update(mask_losses)

        # Offset head
        if self.offset_roi_pool is not None and self.offset_head is not None and self.offset_predictor is not None:
            offset_losses = {}
            if self.training:
                assert targets is not None

                # Use positive proposals for offset head
                num_images = len(proposals)
                all_num_pos = 0
                offset_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos_inds = torch.where(labels[img_id] > 0)[0]
                    all_num_pos += pos_inds.numel()
                    offset_proposals.append(proposals[img_id][pos_inds])
                    pos_matched_idxs.append(matched_idxs[img_id][pos_inds])

                if all_num_pos > 0:
                    # Extract offset features
                    offset_features = self.offset_roi_pool(features, offset_proposals, image_shapes)
                    offset_features = self.offset_head(offset_features)
                    offset_logits = self.offset_predictor(offset_features)

                    # Collect the offset targets
                    offset_targets = []
                    for img_id in range(num_images):
                        pos_inds = torch.where(labels[img_id] > 0)[0]
                        if pos_inds.numel() > 0:
                            matched_idxs_img = pos_matched_idxs[img_id]
                            gt_offsets = targets[img_id]["offsets"][matched_idxs_img]
                            offset_targets.append(gt_offsets)
                    offset_targets = torch.cat(offset_targets, dim=0)
                    offset_logits = offset_logits.view(-1, offset_targets.shape[-1])

                    # Compute offset loss
                    loss_offset = self.compute_offset_loss(offset_logits, offset_targets)
                    offset_losses = {"loss_offset": loss_offset}
                else:
                    offset_losses = {"loss_offset": torch.tensor(0.0, device=features[0].device)}
                losses.update(offset_losses)
            else:
                # Inference
                if len(proposals) > 0:
                    offset_features = self.offset_roi_pool(features, proposals, image_shapes)
                    offset_features = self.offset_head(offset_features)
                    offset_logits = self.offset_predictor(offset_features)

                    # Process offset predictions
                    offset_predictions = self.offset_inference(offset_logits, proposals, image_shapes)

                    # Include offset results in the output
                    for i in range(len(result)):
                        result[i]["offsets"] = offset_predictions[i]
                else:
                    for i in range(len(result)):
                        result[i]["offsets"] = torch.empty((0, 2), device=proposals[0].device)

        return result, losses

    def compute_offset_loss(self, offset_logits, offset_targets):
        # Implement the loss computation for the offset head
        # Here we use L1 loss; adjust as needed
        loss = nn.functional.l1_loss(offset_logits, offset_targets, reduction='mean')
        return loss

    def offset_inference(self, offset_logits, proposals, image_shapes):
        """
        Convert the normalized offset predictions to pixel offsets during inference.

        Args:
            offset_logits (Tensor): The predicted normalized offsets of shape [total_boxes, 2].
            proposals (List[Tensor]): List of proposals (bounding boxes) per image.
            image_shapes (List[Tuple[int, int]]): List of image shapes.

        Returns:
            List[Tensor]: List of offset predictions per image in pixel units.
        """
        offset_predictions = []
        # Split the offset_logits per image
        offset_logits_split = offset_logits.split([len(p) for p in proposals], dim=0)
        for offsets_norm, boxes in zip(offset_logits_split, proposals):
            # Calculate widths and heights of the boxes
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]

            # Avoid division by zero
            widths = torch.clamp(widths, min=1e-6)
            heights = torch.clamp(heights, min=1e-6)

            # Denormalize offsets
            offsets_x = offsets_norm[:, 0] * widths
            offsets_y = offsets_norm[:, 1] * heights
            offsets = torch.stack((offsets_x, offsets_y), dim=1)

            offset_predictions.append(offsets)

        return offset_predictions
