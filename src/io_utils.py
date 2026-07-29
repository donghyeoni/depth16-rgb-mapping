"""I/O utilities: image path configuration and 16-bit depth PNG loading.

The original notebooks hard-coded Google Drive / Colab paths such as
``/content/drive/My Drive/Colab Notebooks/D1_16.png``. Those are replaced here
by a configurable local ``data/`` directory so the code runs anywhere.
"""

import os

import cv2

# Default directory holding the user-supplied 16-bit depth PNGs.
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# The five depth images used by the original project.
DEFAULT_IMAGE_NAMES = [
    "D1_16.png",
    "D2_16.png",
    "D3_16.png",
    "D4_16.png",
    "D5_16.png",
]


def get_image_paths(data_dir=None, names=None):
    """Return the list of full image paths inside ``data_dir``.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the depth PNGs. Defaults to ``<repo>/data``.
    names : list of str, optional
        File names to use. Defaults to ``D1_16.png`` .. ``D5_16.png``.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    names = names or DEFAULT_IMAGE_NAMES
    return [os.path.join(data_dir, name) for name in names]


def load_depth_16bit(image_path):
    """Load a single-channel 16-bit depth image.

    Uses ``cv2.IMREAD_UNCHANGED`` so the full 16-bit range (0..65535) is
    preserved rather than being down-converted to 8-bit.

    Raises
    ------
    FileNotFoundError
        If the image cannot be read from ``image_path``.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image at '{image_path}'. "
            "Place your 16-bit depth PNGs in the data/ directory."
        )
    return image
