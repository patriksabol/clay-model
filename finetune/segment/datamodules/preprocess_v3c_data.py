import os
import sys
from pathlib import Path

import numpy as np
import rasterio as rio


def read_and_chip(ortho_path, mask_path, chip_size, output_dir):
    """
    Reads a pair of GeoTIFF files (ortho and mask), creates chips of specified
    size, and saves them as numpy arrays, skipping black ortho chips.

    Args:
        ortho_path (str or Path): Path to the ortho GeoTIFF file.
        mask_path (str or Path): Path to the mask GeoTIFF file.
        chip_size (int): Size of the square chips.
        output_dir (str or Path): Directory to save the chips.
    """
    os.makedirs(output_dir / "orto", exist_ok=True)
    os.makedirs(output_dir / "masks", exist_ok=True)

    with rio.open(ortho_path) as src_orto, rio.open(mask_path) as src_mask:
        ortho_data = src_orto.read()
        mask_data = src_mask.read()

        n_chips_x = src_orto.width // chip_size
        n_chips_y = src_orto.height // chip_size

        chip_number = 0
        for i in range(n_chips_x):
            for j in range(n_chips_y):
                x1, y1 = i * chip_size, j * chip_size
                x2, y2 = x1 + chip_size, y1 + chip_size

                ortho_chip = ortho_data[:, y1:y2, x1:x2]
                mask_chip = mask_data[:, y1:y2, x1:x2]

                # Check if the ortho chip is not all black
                if not np.all(ortho_chip == 0):
                    ortho_chip_path = os.path.join(
                        output_dir / "orto",
                        f"{Path(ortho_path).stem}_chip_{chip_number}.npy",
                    )
                    mask_chip_path = os.path.join(
                        output_dir / "masks",
                        f"{Path(mask_path).stem}_chip_{chip_number}.npy",
                    )
                    np.save(ortho_chip_path, ortho_chip)
                    np.save(mask_chip_path, mask_chip)
                    chip_number += 1



def process_files(ortho_paths, mask_paths, output_dir, chip_size):
    """
    Processes a list of ortho and mask files, creating chips and saving them,
    skipping pairs if the ortho chip is all black.

    Args:
        ortho_paths (list of Path): List of paths to the ortho GeoTIFF files.
        mask_paths (list of Path): List of paths to the mask GeoTIFF files.
        output_dir (str or Path): Directory to save the chips.
        chip_size (int): Size of the square chips.
    """
    for ortho_path in ortho_paths:
        corresponding_mask_path = next(
            (mask_path for mask_path in mask_paths if mask_path.stem == ortho_path.stem), None
        )

        if corresponding_mask_path:
            print(f"Processing ortho: {ortho_path} and mask: {corresponding_mask_path}")
            read_and_chip(ortho_path, corresponding_mask_path, chip_size, output_dir)
        else:
            print(f"No matching mask found for ortho: {ortho_path}")


def main():
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
    """
    if len(sys.argv) != 4:
        print("Usage: python script.py <data_dir> <output_dir> <chip_size>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    chip_size = int(sys.argv[3])

    train_ortho_paths = list((data_dir / "new_raster_folder" / "train").glob("*.tif"))
    val_ortho_paths = list((data_dir / "new_raster_folder" / "validation").glob("*.tif"))
    train_mask_paths = list((data_dir / "new_mask_raster_folder" / "train").glob("*.tif"))
    val_mask_paths = list((data_dir / "new_mask_raster_folder" / "validation").glob("*.tif"))

    process_files(train_ortho_paths, train_mask_paths, output_dir / "train", chip_size)
    process_files(val_ortho_paths, val_mask_paths, output_dir / "validation", chip_size)


if __name__ == "__main__":
    main()
