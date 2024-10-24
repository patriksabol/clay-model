import argparse
import torch
import os
import yaml
import rasterio
from PIL import Image
from matplotlib import pyplot as plt
from finetune.segment.models.buildings.model import LightingSegmentor  # noqa: F401
from torchvision.transforms import v2


def load_image(image_path):
    """
    Load and preprocess the image for inference.

    Args:
        image_path (str): Path to the input image.

    Returns:
        Tuple[torch.Tensor, dict]: Preprocessed image tensor and georeferencing info.
    """
    # Open the image using rasterio to preserve georeferencing information
    with rasterio.open(image_path) as dataset:
        image = dataset.read([1, 2, 3]).astype('float32')  # Assuming the input is 3 channels
        image = image / 255.0  # Normalize the image to [0, 1]
        transform = dataset.transform
        crs = dataset.crs

    # Apply ImageNet normalization
    preprocess = v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(torch.tensor(image))  # (H, W, C) to (C, H, W)
    return image_tensor.unsqueeze(0), {'transform': transform, 'crs': crs}  # Add batch dimension


def run_inference(model, image_tensor):
    """
    Run inference on a single image.

    Args:
        model (LightingSegmentor): Loaded segmentation model.
        image_tensor (torch.Tensor): Preprocessed image tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Segmentation and regression outputs.
    """
    image_tensor = image_tensor.to(model.device)
    sample = {
        "orto": image_tensor
    }
    with torch.no_grad():
        segmentation_output, regression_output = model(sample)
        segmentation_output = torch.sigmoid(segmentation_output)
        regression_output = regression_output * segmentation_output.shape[-1]  # Scale regression output
    return segmentation_output, regression_output


def save_results(segmentation_output, regression_output, georef_info, save_dir, image_name):
    """
    Save the segmentation and regression outputs as georeferenced TIFF files.

    Args:
        segmentation_output (torch.Tensor): Segmentation output from the model.
        regression_output (torch.Tensor): Regression output from the model.
        georef_info (dict): Georeferencing information (transform and CRS).
        save_dir (str): Directory to save the output.
        image_name (str): Name of the input image (used for naming the output files).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get the georeferencing information
    transform = georef_info['transform']
    crs = georef_info['crs']
    height, width = segmentation_output.shape[-2], segmentation_output.shape[-1]

    # Save segmentation output as a georeferenced TIFF
    seg_image = segmentation_output.squeeze().cpu().numpy()
    with rasterio.open(
            os.path.join(save_dir, f"{image_name}_segmentation.tif"),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,  # Single channel
            dtype=seg_image.dtype,
            crs=crs,
            transform=transform
    ) as dst:
        dst.write(seg_image[0], 1)  # Write the single channel
        dst.write(seg_image[1], 2)  # Write the single channel

    # Save regression output as a georeferenced TIFF (2 channels: x_shift, y_shift)
    reg_image = regression_output.squeeze().cpu().numpy()
    with rasterio.open(
            os.path.join(save_dir, f"{image_name}_regression.tif"),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,  # Two channels for regression (x_shift, y_shift)
            dtype=reg_image.dtype,
            crs=crs,
            transform=transform
    ) as dst:
        dst.write(reg_image[0], 1)  # Write the x_shift channel
        dst.write(reg_image[1], 2)  # Write the y_shift channel


def load_model_from_checkpoint(config, checkpoint_path):
    """
    Load the model from the checkpoint using the configuration file.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        LightingSegmentor: Loaded segmentation model.
    """
    model = LightingSegmentor.load_from_checkpoint(
        checkpoint_path,
        model_arch=config['model']['model_arch'],
        encoder_name=config['model']['encoder_name'],
        encoder_weights=config['model']['encoder_weights'],
        in_channels=config['model']['in_channels'],
        lr=config['model']['lr'],
        wd=config['model']['wd'],
        b1=config['model']['b1'],
        b2=config['model']['b2']
    )
    model.eval()
    return model


def parse_config(config_path):
    """
    Parse the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_image_file(model, image_path, output_dir):
    """
    Process a single image file, run inference, and save the results.

    Args:
        model (LightingSegmentor): Loaded segmentation model.
        image_path (str): Path to the image file.
        output_dir (str): Directory to save the output images.
    """
    # Load and preprocess image along with georeferencing info
    image_tensor, georef_info = load_image(image_path)

    # Run inference
    segmentation_output, regression_output = run_inference(model, image_tensor)

    # Get image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save results as georeferenced TIFFs
    save_results(segmentation_output, regression_output, georef_info, output_dir, image_name)


def main(args):
    # Parse the configuration file
    config = parse_config(args.config)

    # Load the model from checkpoint
    model = load_model_from_checkpoint(config, args.checkpoint)

    # Process each image provided via argparse
    for image_path in args.images:
        process_image_file(model, image_path, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on multiple images using a trained segmentation model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--images', type=str, nargs='+', required=True,
                        help="Paths to input image files (space-separated for multiple images).")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save output images.")

    args = parser.parse_args()
    main(args)

    print("Inference completed!")
