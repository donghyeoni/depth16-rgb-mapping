"""Degradation operators used in Project 2: Gaussian noise and JPEG round-trip.

These simulate a realistic transport/storage pipeline where the encoded RGB
image is perturbed by sensor noise and lossy JPEG re-compression before being
reconstructed.
"""

import cv2
import numpy as np


def add_gaussian_noise(image, std=50, mean=0):
    """Add clipped Gaussian noise to an 8-bit image.

    The noise is drawn with the given ``mean`` and standard deviation ``std``,
    added in int16 space, then clipped to 0..255 and cast back to uint8.
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def jpeg_roundtrip(image, quality=30):
    """Encode ``image`` to JPEG at the given quality (QP) and decode it back.

    Uses ``cv2.imencode`` / ``cv2.imdecode`` so nothing touches disk. The
    ``quality`` parameter is the OpenCV JPEG quality (0..100); lower means more
    lossy compression.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, compressed = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        raise RuntimeError("cv2.imencode failed to JPEG-encode the image.")
    decoded = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
    return decoded
