#!/usr/bin/env python3
"""
dataset_generator.py  —  Drift-Sense Synthetic Dataset Generator
=================================================================
Generates synthetic SEM image pairs (Reference + Search) for the
Drift-Sense Navigation-Error Recovery challenge by Applied Materials,
across four explicit, labelled difficulty tiers:

  easy       periodic array + STRONG local landmark, low noise
  medium     periodic array + WEAK local landmark, medium noise
  hard       PURE periodic array (no landmark), low/medium noise
  very_hard  PURE periodic array + heavy noise/blur/drift

"hard"/"very_hard" deliberately reproduce the documented periodic-mat-
block ambiguity failure mode (see README.md failure analysis): a
reference crop with no locally-unique content. Every row also records an
FFT-autocorrelation periodicity_score for the reference crop and whether
a landmark actually landed inside it, so evaluation can be conditioned on
*why* a pair is hard, not just which noise bucket it came from.

Physical setup (fixed by problem statement)
-------------------------------------------
  Reference image : 1000x1000 px @ 1 nm/px (1 um FOV, 100x mag)
  Search image    : 1000x1000 px @ 10 nm/px (10 um FOV, 10x mag)
  Scale ratio     : exactly 10x

Architecture styles
-------------------
  DRAM   : periodic word-line / bit-line grid with storage contacts
  FinFET : dense vertical fin lines crossed by horizontal gate bars

SEM physics implemented (unchanged, see citations.md)
-------------------------------------------------------
  Edge brightening, Gaussian PSF beam blur + astigmatism, Poisson shot
  noise, Gaussian detector noise, raster-scan drift, barrel/pincushion
  distortion, surface charging streaks, multiplicative speckle noise,
  salt-and-pepper impulse noise, vignetting, gamma nonlinearity, pattern
  collapse/gap bridging, missing contacts/vias.

Usage
-----
  # Generate 40 pairs across all four tiers and all architectures
  python dataset_generator.py --num-samples 40 --output-dir ./dataset

  # Only the hardest tiers, DRAM-dense/FinFET-7nm only
  python dataset_generator.py --num-samples 20 --tiers hard very_hard \\
      --architectures dram_dense finfet_7nm --output-dir ./dataset_hard

Output layout
-------------
  <output-dir>/
    train/
      reference/  00000.png ... 00N.png
      search/     00000.png ... 00N.png
      manifest.csv  (id, paths, gt_x, gt_y, gt_box, tier, landmark/
                      periodicity diagnostics, architecture, all SEM params)
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import GenerationParams, generate_sample_tiered, TIER_NAMES
from src.presets import PRESETS


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=40,
                   help="Total pairs, split round-robin across --tiers x --architectures")
    p.add_argument("--tiers", nargs="+", default=TIER_NAMES, choices=TIER_NAMES)
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()),
                   choices=list(PRESETS.keys()))
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="./dataset")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    base_params = GenerationParams()

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(split_dir, "manifest.csv")
    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "architecture", "tier", "seed",
        "landmark_count_total", "landmark_in_crop", "landmark_strength",
        "landmark_ref_x", "landmark_ref_y", "landmark_size",
        "on_boundary", "periodicity_score",
    ]

    # round-robin tier x architecture so the split is balanced and reproducible
    combos = [(t, a) for t in args.tiers for a in args.architectures]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i in range(args.num_samples):
            tier, arch = combos[i % len(combos)]
            sample = generate_sample_tiered(arch, tier, rng, base_params)

            ref_path = os.path.join(ref_dir, f"{i:05d}.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")
            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])

            gx0, gy0, gw, gh = sample["gt_box"]
            writer.writerow({
                "id": i, "reference_path": ref_path, "search_path": search_path,
                "gt_x": sample["gt_x"], "gt_y": sample["gt_y"],
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture": arch, "tier": tier, "seed": args.seed,
                "landmark_count_total": sample["landmark_count_total"],
                "landmark_in_crop": sample["landmark_in_crop"],
                "landmark_strength": sample["landmark_strength"],
                "landmark_ref_x": sample["landmark_ref_x"],
                "landmark_ref_y": sample["landmark_ref_y"],
                "landmark_size": sample["landmark_size"],
                "on_boundary": sample["on_boundary"],
                "periodicity_score": f"{sample['periodicity_score']:.4f}",
            })
            print(f"[{i + 1:3d}/{args.num_samples}] {arch:15s} tier={tier:10s} "
                  f"periodicity={sample['periodicity_score']:.3f} "
                  f"landmark_in_crop={sample['landmark_in_crop']} "
                  f"gt=({sample['gt_x']:.1f},{sample['gt_y']:.1f})")

    print(f"\nWrote {args.num_samples} samples to {split_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
