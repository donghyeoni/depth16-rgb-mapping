"""Project 1: clean mapping (A) and reconstruction + MSE (B).

For each of the three methods this script encodes every 16-bit depth image
into an 8-bit-per-channel RGB representation, reconstructs the 16-bit image,
and reports the MSE. No degradation is applied.

Usage
-----
    python experiments/01_mapping_clean.py --data-dir data --show
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import get_image_paths, load_depth_16bit
from src.mapping import (
    byte_split,
    map_16bit_to_rgb,
    map_image_to_rgb,
    normalize_to_1535,
)
from src.metrics import calculate_mse
from src.reconstruction import (
    byte_recombine,
    denormalize_from_1535,
    map_rgb_to_16bit,
    rgb_to_normalized,
)
from src import visualize


def run_method_1(image, show=False, index=0):
    """Byte split -> upper-8-bit reconstruction."""
    upper_8bit, lower_8bit = byte_split(image)
    restored = byte_recombine(upper_8bit)
    mse = calculate_mse(image, restored)
    if show:
        visualize.show_byte_split(image, upper_8bit, lower_8bit, index)
        visualize.show_reconstruction(restored, mse, index)
    return mse


def run_method_2(image, show=False, index=0):
    """Spectral color-wheel map -> inverse map."""
    normalized = normalize_to_1535(image)
    rgb_image = map_image_to_rgb(normalized)
    restored_norm = rgb_to_normalized(rgb_image, clamp_unknown=False)
    restored = denormalize_from_1535(restored_norm)
    mse = calculate_mse(image, restored)
    if show:
        visualize.show_rgb_mapping(image, rgb_image, index, normalized)
        visualize.show_reconstruction(restored, mse, index)
    return mse


def run_method_3(image, show=False, index=0):
    """6/6/4 bit-field split -> recombine."""
    rgb_image = map_16bit_to_rgb(image)
    restored = map_rgb_to_16bit(rgb_image)
    mse = calculate_mse(image, restored)
    if show:
        visualize.show_rgb_mapping(image, rgb_image, index)
        visualize.show_reconstruction(restored, mse, index)
    return mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory holding D1_16.png .. D5_16.png (default: <repo>/data).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib comparison figures.",
    )
    args = parser.parse_args()

    paths = get_image_paths(args.data_dir)
    methods = [
        ("Method 1 (byte split)", run_method_1),
        ("Method 2 (spectral wheel)", run_method_2),
        ("Method 3 (6/6/4 bit-field)", run_method_3),
    ]

    for name, func in methods:
        print(f"\n=== {name} ===")
        for idx, path in enumerate(paths):
            image = load_depth_16bit(path)
            mse = func(image, show=args.show, index=idx)
            print(f"  {os.path.basename(path)}: MSE = {mse:.2f}")


if __name__ == "__main__":
    main()
