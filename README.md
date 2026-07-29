# depth16-rgb-mapping

Encode a single-channel **16-bit depth image** into an **8-bit-per-channel RGB**
image, reconstruct the original depth from that RGB, and measure the fidelity of
the round-trip with **Mean Squared Error (MSE)**.

The motivation is practical: many image formats, codecs, and display/transport
pipelines are built for 8-bit RGB, not 16-bit single-channel data. Packing a
16-bit depth map into three 8-bit channels lets it flow through those pipelines;
the question this project studies is *how much precision is lost* — first in a
clean round-trip, then under realistic degradation (sensor noise and lossy JPEG
re-compression).

## Overview

* **Project 1 — clean pipeline**
  * *(A)* Forward mapping: 16-bit depth -> RGB, with visualizations.
  * *(B)* Reconstruction: RGB -> 16-bit depth, with MSE.
* **Project 2 — degraded pipeline**
  * *(A)* Insert Gaussian noise and a lossy JPEG encode/decode round-trip
    between encoding and reconstruction, then measure MSE.
  * *(B)* Histogram analysis of the depth distributions plus a per-image
    maximum-normalization variant of the spectral method.

## The three mapping methods

Each method is a forward mapping (in `src/mapping.py`) paired with its exact
inverse (in `src/reconstruction.py`).

### Method 1 — Byte split
Split the 16-bit value into two 8-bit planes: the upper byte (`value >> 8`) and
the lower byte (`value & 0xFF`). Reconstruction in the original project uses the
**upper byte only** (`upper << 8`), discarding the lower byte — a deliberately
lossy baseline that keeps the coarse depth structure.

### Method 2 — Spectral color-wheel map
Normalize the depth to the range `0..1535`, then map each value onto a piecewise
spectral color wheel across six 256-wide segments
(red -> yellow -> green -> cyan -> blue -> magenta). The inverse (`map_to_value`)
identifies the segment from the RGB triple and recovers the normalized value,
which is then rescaled back to the 16-bit range. This spreads depth across all
three channels as a smooth, human-readable color gradient.

### Method 3 — 6/6/4 bit-field split
Partition the 16 bits into three fields written to the color channels:
upper 6 bits -> R, middle 6 bits -> G, lower 4 bits -> B
(`R = (v>>10)&0x3F`, `G = (v>>4)&0x3F`, `B = v&0x0F`). Reconstruction recombines
them (`(R<<10)|(G<<4)|B`). In a *clean* round-trip this is lossless; under JPEG
it is sensitive because compression corrupts the least-significant bits.

## Dataset

The project uses **five 16-bit grayscale depth PNGs**, named
`D1_16.png` .. `D5_16.png`, loaded locally with `cv2.IMREAD_UNCHANGED` so the
full 0..65535 range is preserved.

These images are **not included** in this repository. Supply your own 16-bit
depth PNGs by placing them in the `data/` directory (see `data/README.md`). The
original notebooks loaded them from Google Drive in Colab; those hard-coded
paths have been replaced by a configurable local directory and `--data-dir`.

## Repository structure

```
depth16-rgb-mapping/
├── src/
│   ├── io_utils.py         # image path config + 16-bit PNG loading (IMREAD_UNCHANGED)
│   ├── mapping.py          # 3 forward mappings: byte_split, map_to_rgb, map_16bit_to_rgb
│   ├── reconstruction.py   # 3 inverses: byte_recombine, map_to_value, map_rgb_to_16bit
│   ├── degradation.py      # add_gaussian_noise, jpeg_roundtrip (cv2 encode/decode)
│   ├── metrics.py          # calculate_mse
│   └── visualize.py        # matplotlib comparison grids / histograms
├── experiments/
│   ├── 01_mapping_clean.py     # Project 1: clean mapping + reconstruction + MSE
│   └── 02_mapping_degraded.py  # Project 2: degradation (noise + JPEG) + MSE + histogram
├── notebooks/              # original Colab notebooks (unmodified)
├── docs/                   # project report (PDF)
├── data/                   # user-supplied depth PNGs (git-ignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

Requires: `numpy`, `opencv-python`, `matplotlib`.

## Usage

Put `D1_16.png` .. `D5_16.png` in `data/`, then:

```bash
# Project 1: clean round-trip, print MSE for all three methods
python experiments/01_mapping_clean.py

# add --show to display the comparison / reconstruction figures
python experiments/01_mapping_clean.py --show

# Project 2: degraded round-trip (Gaussian noise + lossy JPEG)
python experiments/02_mapping_degraded.py --jpeg-quality 30 --noise-std 50

# Project 2 (B): also run the histogram / per-image max normalization variant
python experiments/02_mapping_degraded.py --histogram --show

# use a different image directory
python experiments/01_mapping_clean.py --data-dir /path/to/my/depth/pngs
```

## Notes

* **Faithful port.** The three algorithms are preserved exactly from the
  original Colab notebooks (kept unmodified under `notebooks/`). The code was
  refactored into modules; the mapping math was not changed.
* **Integer dtype corrections.** Three small dtype fixes relative to the
  notebooks, none of which change the algorithms — they only prevent overflow
  under modern NumPy (2.x): `calculate_mse` casts to `float64` before
  differencing; `map_to_value` casts channels to Python `int` before its
  `1280 + ...` arithmetic; and `map_rgb_to_16bit` casts channels to `int`
  before the `<< 10` shift (a `uint8` shift would wrap to 0). With these,
  Method 3's clean round-trip is exactly lossless (MSE = 0) as intended.
* **Method 2 inverse behaviour.** In the clean pipeline, RGB triples that match
  no color-wheel segment are treated as `None`; under degradation
  (`clamp_unknown=True`) they fall back to `0`, matching the notebooks where
  noise/JPEG can push colors out of the exact gamut.
* **Performance.** Methods 2 and 3 iterate pixel-by-pixel in Python, exactly as
  in the notebooks, so large images run slowly. The algorithms are vectorizable
  if speed matters.
* **Report.** See `docs/Performance Analysis of RGB Channel Mapping and
  Reconstruction for 16-bit Depth Images.pdf` for the full write-up and results.
