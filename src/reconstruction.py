"""Inverse mappings: reconstruct a 16-bit depth image from its RGB encoding.

Each function inverts the corresponding forward mapping in ``mapping.py``:

* Method 1 -- ``byte_recombine``: shift the upper 8 bits back into place.
* Method 2 -- ``map_to_value`` / ``rgb_to_normalized``: invert the spectral
  color-wheel map, then rescale from 0..1535 back to the 16-bit range.
* Method 3 -- ``map_rgb_to_16bit``: recombine the 6/6/4 bit fields.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Method 1: byte recombine (upper-8-bit reconstruction)
# ---------------------------------------------------------------------------
def byte_recombine(upper_8bit):
    """Reconstruct a 16-bit image from its upper 8 bits.

    Mirrors the original notebook, which reconstructed depth from the upper
    byte only: ``restored = upper_8bit << 8``. The lower byte is discarded, so
    this is a lossy reconstruction.
    """
    return (upper_8bit.astype(np.uint16) << 8)


# ---------------------------------------------------------------------------
# Method 2: inverse spectral color-wheel map
# ---------------------------------------------------------------------------
def map_to_value(r, g, b, clamp_unknown=False):
    """Invert :func:`mapping.map_to_rgb` for a single (R, G, B) triple.

    Parameters
    ----------
    clamp_unknown : bool
        When True, triples that match no segment return 0 (the behaviour used
        in the degraded Project 2 experiment, where noise/JPEG can produce
        out-of-gamut colors). When False, such triples return ``None`` (the
        clean Project 1 behaviour, which assumes exact colors).
    """
    # Cast to Python ints: NumPy uint8 channel values would overflow on the
    # ``1280 + ...`` style arithmetic below under NumPy 2.x.
    r, g, b = int(r), int(g), int(b)
    if r == 255 and 0 <= g < 256 and b == 0:
        return g
    elif 0 <= r < 256 and g == 255 and b == 0:
        return 256 + (255 - r)
    elif r == 0 and g == 255 and 0 <= b < 256:
        return 512 + b
    elif r == 0 and 0 <= g < 256 and b == 255:
        return 768 + (255 - g)
    elif 0 <= r < 256 and g == 0 and b == 255:
        return 1024 + r
    elif r == 255 and g == 0 and 0 <= b < 256:
        return 1280 + (255 - b)
    else:
        return 0 if clamp_unknown else None


def rgb_to_normalized(rgb_image, clamp_unknown=False):
    """Invert the spectral map for a full RGB image, returning 0..1535 values."""
    height, width = rgb_image.shape[:2]
    restored = np.zeros((height, width), dtype=np.uint16)
    for i in range(height):
        for j in range(width):
            r, g, b = rgb_image[i, j]
            value = map_to_value(r, g, b, clamp_unknown=clamp_unknown)
            restored[i, j] = value if value is not None else 0
    return restored


def denormalize_from_1535(normalized_image, max_value=65535):
    """Rescale a 0..1535 image back to the 16-bit range."""
    return (normalized_image / 1535 * max_value).astype(np.uint16)


# ---------------------------------------------------------------------------
# Method 3: recombine 6/6/4 bit fields
# ---------------------------------------------------------------------------
def map_rgb_to_16bit(rgb_image):
    """Invert :func:`mapping.map_16bit_to_rgb` by recombining the bit fields."""
    height, width, _ = rgb_image.shape
    restored_image = np.zeros((height, width), dtype=np.uint16)

    for i in range(height):
        for j in range(width):
            # Cast to Python ints: a uint8 left-shift by 10 wraps to 0.
            r_ch = int(rgb_image[i, j, 0])
            g_ch = int(rgb_image[i, j, 1])
            b_ch = int(rgb_image[i, j, 2])

            # For clean input the channels hold 6/6/4-bit fields, so the packed
            # value fits in 16 bits. Degraded (noisy) input can push it past
            # 65535; mask to 16 bits to preserve the original wrap-around
            # behaviour (NumPy < 2 wrapped silently on assignment; NumPy >= 2
            # raises OverflowError instead).
            restored_image[i, j] = ((r_ch << 10) | (g_ch << 4) | b_ch) & 0xFFFF

    return restored_image
