"""Forward mappings: encode a 16-bit depth image into an 8-bit-per-channel RGB.

Three methods, preserved exactly from the original notebooks:

* Method 1 -- ``byte_split``: split the 16-bit value into its upper 8 bits
  (``>> 8``) and lower 8 bits (``& 0xFF``).
* Method 2 -- ``map_to_rgb`` / ``map_image_to_rgb``: normalize the depth to the
  0..1535 range and map each value onto a piecewise spectral color wheel.
* Method 3 -- ``map_16bit_to_rgb``: split the 16-bit value into a 6/6/4 bit
  field written to the R/G/B channels.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Method 1: byte split
# ---------------------------------------------------------------------------
def byte_split(image_16bit):
    """Split a 16-bit image into (upper_8bit, lower_8bit) uint8 planes."""
    upper_8bit = (image_16bit >> 8).astype(np.uint8)
    lower_8bit = (image_16bit & 0xFF).astype(np.uint8)
    return upper_8bit, lower_8bit


# ---------------------------------------------------------------------------
# Method 2: normalize to 0..1535 then piecewise spectral color-wheel map
# ---------------------------------------------------------------------------
def normalize_to_1535(image_16bit, max_value=65535):
    """Normalize a 16-bit image to the 0..1535 integer range.

    ``max_value`` defaults to 65535 (full 16-bit range). The Project 2 (B)
    histogram variant instead used a per-image maximum; pass it explicitly.
    """
    return (image_16bit / max_value * 1535).astype(np.uint16)


def map_to_rgb(value):
    """Map a single normalized value (0..1535) to an (R, G, B) tuple.

    The six 256-wide segments trace a spectral color wheel:
    red -> yellow -> green -> cyan -> blue -> magenta.
    """
    if 0 <= value < 256:
        return (255, value, 0)                    # (255, 0~255, 0)
    elif 256 <= value < 512:
        return (255 - (value - 256), 255, 0)      # (255~0, 255, 0)
    elif 512 <= value < 768:
        return (0, 255, value - 512)              # (0, 255, 0~255)
    elif 768 <= value < 1024:
        return (0, 255 - (value - 768), 255)      # (0, 255~0, 255)
    elif 1024 <= value < 1280:
        return (value - 1024, 0, 255)             # (0~255, 0, 255)
    elif 1280 <= value < 1536:
        return (255, 0, 255 - (value - 1280))     # (255, 0, 255~0)


def map_image_to_rgb(normalized_image):
    """Apply :func:`map_to_rgb` to every pixel of a normalized image."""
    rgb_image = np.zeros((*normalized_image.shape, 3), dtype=np.uint8)
    for i in range(normalized_image.shape[0]):
        for j in range(normalized_image.shape[1]):
            rgb_image[i, j] = map_to_rgb(normalized_image[i, j])
    return rgb_image


# ---------------------------------------------------------------------------
# Method 3: 6/6/4 bit-field split into R/G/B
# ---------------------------------------------------------------------------
def map_16bit_to_rgb(image_16bit):
    """Split each 16-bit value into 6/6/4 bits written to R/G/B channels."""
    height, width = image_16bit.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel_value = image_16bit[i, j]

            r_ch = (pixel_value >> 10) & 0x3F      # upper 6 bits
            g_ch = (pixel_value >> 4) & 0x3F       # middle 6 bits
            b_ch = pixel_value & 0x0F              # lower 4 bits

            rgb_image[i, j, 0] = r_ch              # R channel
            rgb_image[i, j, 1] = g_ch              # G channel
            rgb_image[i, j, 2] = b_ch              # B channel

    return rgb_image
