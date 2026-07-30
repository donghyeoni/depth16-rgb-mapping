"""Regenerate every committed artifact under ``results/`` in one command.

The original notebooks read five 16-bit depth PNGs (``D1_16.png`` ..
``D5_16.png``) captured from a real sensor; those are not redistributed. To make
the pipeline reproducible with **no external data**, this script synthesizes
five deterministic 16-bit depth maps (fixed seeds) into ``results/synthetic_data/``
and runs both experiments on them:

* ``results/01_mapping_clean.log``      -- MSE of each encode/decode method
* ``results/02_mapping_degraded.log``   -- MSE under noise + JPEG round-trip
* ``results/mse_summary.json``          -- parsed MSE table

The original notebook figures are preserved under
``results/notebook_reference/``.

Usage
-----
    python run_all.py
"""

import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(OUT_DIR, "synthetic_data")


def make_depth_16bit(size=128, seed=0):
    """A deterministic single-channel 16-bit depth map with smooth ramps and
    a few raised planar regions (structured depth, not pure noise)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    base = (xx + yy) / (2 * size) * 40000.0  # smooth diagonal depth ramp
    for _ in range(4):
        cx, cy = rng.integers(0, size, 2)
        rad = int(rng.integers(15, 40))
        depth = float(rng.integers(5000, 60000))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        base[mask] = depth
    base += rng.normal(0, 300, base.shape)
    return np.clip(base, 0, 65535).astype(np.uint16)


def run(name, args):
    log_path = os.path.join(OUT_DIR, f"{name}.log")
    print(f"  {name} ...")
    proc = subprocess.run([sys.executable] + args, cwd=REPO_ROOT,
                          capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n" + proc.stderr)
    return proc.stdout


def parse_mse(log_text):
    """Parse '=== <section> ===' blocks and '<file>: MSE = <x>' lines."""
    summary, section = {}, None
    for line in log_text.splitlines():
        m = re.match(r"=== (.+?) ===", line.strip())
        if m:
            section = m.group(1)
            summary[section] = {}
        elif section:
            m = re.match(r"\s*(\S+):\s*MSE\s*=\s*([-\d.eE]+)", line)
            if m:
                summary[section][m.group(1)] = float(m.group(2))
    return summary


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"

    for i in range(5):
        cv2.imwrite(os.path.join(DATA_DIR, f"D{i + 1}_16.png"),
                    make_depth_16bit(seed=i))
    print(f"5 synthetic 16-bit depth maps written to "
          f"{os.path.relpath(DATA_DIR, REPO_ROOT)}")

    clean = run("01_mapping_clean",
                ["experiments/01_mapping_clean.py", "--data-dir", DATA_DIR])
    degraded = run("02_mapping_degraded",
                   ["experiments/02_mapping_degraded.py", "--data-dir", DATA_DIR,
                    "--histogram"])

    summary = {"clean": parse_mse(clean), "degraded": parse_mse(degraded)}
    with open(os.path.join(OUT_DIR, "mse_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Artifacts under results/.")


if __name__ == "__main__":
    main()
