#!/usr/bin/env python3
"""
add_predictions.py  —  Write localizer predictions into the manifest
========================================================================
The problem statement's checklist requires the CSV/manifest to contain
"paths, true coordinates, predictions and metadata." dataset_generator.py
writes ground truth (gt_x/gt_y) at generation time, but predictions only
exist if you run localize.py -- and nothing writes them back per-sample.
This closes that gap: runs localize() over every row in a manifest and
writes a new CSV with the original columns plus predicted x/y, error,
method, score, and latency.

Does NOT modify the original manifest.csv (avoids any risk of corrupting
the ground-truth record) -- writes a new file alongside it instead.

Usage
-----
  python add_predictions.py --manifest ./dataset/train/manifest.csv \\
      --output ./dataset/train/manifest_with_predictions.csv
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from localize import localize


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="./dataset/train/manifest.csv")
    ap.add_argument("--output", default=None,
                     help="Default: <manifest_dir>/manifest_with_predictions.csv")
    ap.add_argument("--tolerance-px", type=float, default=5.0)
    args = ap.parse_args()

    if args.output is None:
        manifest_dir = os.path.dirname(args.manifest) or "."
        args.output = os.path.join(manifest_dir, "manifest_with_predictions.csv")

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    if not rows:
        raise SystemExit(f"No rows found in {args.manifest}")

    out_fieldnames = list(rows[0].keys()) + [
        "pred_x", "pred_y", "error_px", "correct_at_tolerance",
        "method", "score", "latency_ms",
    ]

    n_correct = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i, row in enumerate(rows):
            ref = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
            srch = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
            if ref is None or srch is None:
                print(f"  [skip] missing image for row {row.get('id', i)}: {row['reference_path']}")
                continue

            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
            t0 = time.perf_counter()
            result = localize(ref, srch)
            lat_ms = (time.perf_counter() - t0) * 1e3

            err = float(np.hypot(result["x"] - gt_x, result["y"] - gt_y))
            correct = err <= args.tolerance_px
            n_correct += int(correct)

            out_row = dict(row)
            out_row.update({
                "pred_x": f"{result['x']:.4f}",
                "pred_y": f"{result['y']:.4f}",
                "error_px": f"{err:.4f}",
                "correct_at_tolerance": correct,
                "method": result["method"],
                "score": f"{result['score']:.4f}",
                "latency_ms": f"{lat_ms:.1f}",
            })
            writer.writerow(out_row)

            marker = "OK" if correct else "FAIL"
            print(f"  [{i+1:3d}/{len(rows)}] id={row.get('id', i):>3} "
                  f"{row.get('architecture',''):15s} err={err:8.2f}px {marker:4s} "
                  f"method={result['method']}")

    print(f"\nAccuracy @{args.tolerance_px:.0f}px: {n_correct}/{len(rows)} "
          f"({100.0 * n_correct / len(rows):.1f}%)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
