"""Project 2: mapping + reconstruction under degradation (A) and analysis (B).

Repeats the three encoding methods but inserts a degradation stage (Gaussian
noise and/or a lossy JPEG re-compression round-trip) between encoding and
reconstruction, then reports the MSE. Part B adds a histogram / per-image
maximum-normalization variant.

Usage
-----
    python experiments/02_mapping_degraded.py --jpeg-quality 30 --noise-std 50 --show
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.degradation import add_gaussian_noise, jpeg_roundtrip
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
    map_to_value,
    rgb_to_normalized,
)
from src import visualize


def run_method_1(image, jpeg_quality=90, show=False, index=0):
    """Byte split -> JPEG round-trip on the upper byte -> reconstruction."""
    upper_8bit, _ = byte_split(image)
    decoded = jpeg_roundtrip(upper_8bit, quality=jpeg_quality)
    restored = byte_recombine(decoded)
    mse = calculate_mse(image, restored)
    if show:
        visualize.show_reconstruction(restored, mse, index)
    return mse


def run_method_2(image, jpeg_quality=30, show=False, index=0):
    """Spectral map -> JPEG round-trip on RGB -> inverse map (clamped)."""
    normalized = normalize_to_1535(image)
    rgb_image = map_image_to_rgb(normalized)
    decoded = jpeg_roundtrip(rgb_image, quality=jpeg_quality)

    restored_norm = rgb_to_normalized(decoded, clamp_unknown=True)
    restored = denormalize_from_1535(restored_norm)
    mse = calculate_mse(image, restored)
    if show:
        visualize.show_reconstruction(restored, mse, index)
    return mse


def run_method_3(image, jpeg_quality=30, show=False, index=0):
    """6/6/4 bit-field split -> JPEG round-trip on RGB -> recombine."""
    rgb_image = map_16bit_to_rgb(image)
    decoded = jpeg_roundtrip(rgb_image, quality=jpeg_quality)
    restored = map_rgb_to_16bit(decoded)
    mse = calculate_mse(image, restored)
    if show:
        visualize.show_reconstruction(restored, mse, index)
    return mse


def run_histogram_variant(image, noise_std=10, jpeg_quality=90, show=False, index=0):
    """Project 2 (B): per-image max normalization + noise + JPEG (Method 2).

    Instead of normalizing against the full 16-bit range, this variant
    normalizes against the image's own maximum pixel value, and reconstructs
    from the noisy RGB image using the robust (max/min-snapping) inverse map.
    """
    if show:
        visualize.show_histogram(image, index)

    max_value = int(np.max(image))
    if max_value == 0:
        max_value = 1  # guard against all-zero images

    normalized = (image / max_value * 1535).astype(np.uint16)
    rgb_image = map_image_to_rgb(normalized)

    noisy_rgb = add_gaussian_noise(rgb_image, std=noise_std)
    # JPEG round-trip is computed to mirror the notebook pipeline.
    _ = jpeg_roundtrip(rgb_image, quality=jpeg_quality)

    restored = np.zeros(normalized.shape, dtype=np.uint16)
    for i in range(noisy_rgb.shape[0]):
        for j in range(noisy_rgb.shape[1]):
            r, g, b = noisy_rgb[i, j]
            # Snap the noisy channels to the segment extremes before inversion.
            mn, mx = min(r, g, b), max(r, g, b)
            r = 255 if r == mx else (0 if r == mn else r)
            g = 255 if g == mx else (0 if g == mn else g)
            b = 255 if b == mx else (0 if b == mn else b)
            value = map_to_value(r, g, b, clamp_unknown=True)
            restored[i, j] = value if value is not None else 0

    restored = (restored / 1535 * max_value).astype(np.uint16)
    mse = calculate_mse(image, restored)
    if show:
        visualize.show_reconstruction(restored, mse, index)
    return mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None,
                        help="Directory holding the depth PNGs (default: <repo>/data).")
    parser.add_argument("--jpeg-quality", type=int, default=30,
                        help="OpenCV JPEG quality / QP for the RGB round-trip (0-100).")
    parser.add_argument("--noise-std", type=int, default=50,
                        help="Gaussian noise standard deviation.")
    parser.add_argument("--histogram", action="store_true",
                        help="Also run the Project 2 (B) histogram / per-image max variant.")
    parser.add_argument("--show", action="store_true",
                        help="Display matplotlib figures.")
    args = parser.parse_args()

    paths = get_image_paths(args.data_dir)
    methods = [
        ("Method 1 (byte split, QP=90)",
         lambda im, i: run_method_1(im, jpeg_quality=90, show=args.show, index=i)),
        ("Method 2 (spectral wheel)",
         lambda im, i: run_method_2(im, jpeg_quality=args.jpeg_quality, show=args.show, index=i)),
        ("Method 3 (6/6/4 bit-field)",
         lambda im, i: run_method_3(im, jpeg_quality=args.jpeg_quality, show=args.show, index=i)),
    ]

    for name, func in methods:
        print(f"\n=== {name} ===")
        for idx, path in enumerate(paths):
            image = load_depth_16bit(path)
            mse = func(image, idx)
            print(f"  {os.path.basename(path)}: MSE = {mse:.2f}")

    if args.histogram:
        print("\n=== Project 2 (B): per-image max normalization variant ===")
        for idx, path in enumerate(paths):
            image = load_depth_16bit(path)
            mse = run_histogram_variant(
                image, noise_std=args.noise_std,
                jpeg_quality=args.jpeg_quality, show=args.show, index=idx,
            )
            print(f"  {os.path.basename(path)}: MSE = {mse:.2f}")


if __name__ == "__main__":
    main()
