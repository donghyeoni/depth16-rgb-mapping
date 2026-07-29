"""Matplotlib helpers for comparison grids and histograms."""

import matplotlib.pyplot as plt


def show_byte_split(original, upper_8bit, lower_8bit, index=0):
    """Show original / upper-8-bit / lower-8-bit for Method 1 (forward)."""
    plt.figure(figsize=(30, 10))

    plt.subplot(1, 3, 1)
    plt.title(f"Original Image {index + 1}")
    plt.imshow(original, cmap="gray", vmin=0, vmax=65535)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Upper 8-bit Image {index + 1}")
    plt.imshow(upper_8bit, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Lower 8-bit Image {index + 1}")
    plt.imshow(lower_8bit, cmap="gray")
    plt.axis("off")

    plt.show()


def show_rgb_mapping(original, rgb_image, index=0, normalized=None):
    """Show original (and optional normalized) alongside the RGB encoding."""
    if normalized is not None:
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.title(f"Original Image {index + 1}")
        plt.imshow(original, cmap="gray", vmin=0, vmax=65535)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"Normalized (0-1535) Image {index + 1}")
        plt.imshow(normalized, cmap="gray", vmin=0, vmax=1535)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title(f"RGB Image {index + 1}")
        plt.imshow(rgb_image)
        plt.axis("off")
    else:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image {index + 1}")
        plt.imshow(original, cmap="gray", vmin=0, vmax=65535)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"RGB Image {index + 1}")
        plt.imshow(rgb_image)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_reconstruction(restored, mse_value, index=0):
    """Show a reconstructed 16-bit image annotated with its MSE."""
    plt.figure(figsize=(10, 5))
    plt.title(f"Restored Image {index + 1}\nMSE: {mse_value:.2f}")
    plt.imshow(restored, cmap="gray", vmin=0, vmax=65535)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_histogram(depth_image, index=0, bins=65536, value_range=(0, 65535)):
    """Plot the pixel-value histogram of a depth image (Project 2 B)."""
    plt.figure(figsize=(10, 5))
    plt.hist(
        depth_image.ravel(),
        bins=bins,
        range=value_range,
        color="blue",
        histtype="step",
    )
    plt.ylim(0, 1000)
    plt.title(f"D{index + 1}_16.png")
    plt.xlabel("Pixel Value")
    plt.ylabel("Number")
    plt.xlim(list(value_range))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
