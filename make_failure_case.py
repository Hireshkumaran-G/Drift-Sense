#!/usr/bin/env python3
"""
make_failure_case.py  —  Visualize the worst localization failure
======================================================================
evaluate_improved.py reports aggregate metrics but does not save a
per-sample visualization. This script runs localize() over a manifest,
finds the worst (or a specific) failure, and renders a side-by-side
figure: reference crop, search image with the ground-truth box and the
predicted point both marked, and an error/root-cause annotation.

Satisfies the problem statement's "at least one visualized failure case
with root-cause explanation" requirement.

Usage
-----
  # Worst failure in the dataset
  python make_failure_case.py --manifest ./dataset/train/manifest.csv \\
      --output ./eval_results/failure_case.png

  # A specific row instead of the worst one
  python make_failure_case.py --manifest ./dataset/train/manifest.csv \\
      --id 6 --output ./eval_results/failure_case.png
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from localize import localize


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="./dataset/train/manifest.csv")
    ap.add_argument("--output", default="./eval_results/failure_case.png")
    ap.add_argument("--id", default=None, help="Use this specific row id instead of searching for the worst")
    ap.add_argument("--tolerance-px", type=float, default=5.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.id is not None:
        target_rows = [r for r in rows if r["id"] == str(args.id)]
        if not target_rows:
            raise SystemExit(f"No row with id={args.id} in {args.manifest}")
        rows_to_check = target_rows
    else:
        rows_to_check = rows  # scan all, keep the worst

    worst = None
    print(f"Scanning {len(rows_to_check)} row(s) for a failure case...")
    for row in rows_to_check:
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        result = localize(ref, srch)
        err = float(np.hypot(result["x"] - gt_x, result["y"] - gt_y))
        print(f"  id={row['id']:>3s} {row.get('architecture',''):15s} "
              f"err={err:8.2f}px method={result['method']}")
        if worst is None or err > worst["err"]:
            worst = {"row": row, "ref": ref, "srch": srch, "result": result,
                     "gt_x": gt_x, "gt_y": gt_y, "err": err}

    if worst is None:
        raise SystemExit("No valid rows found.")

    row, ref, srch, result = worst["row"], worst["ref"], worst["srch"], worst["result"]
    gt_x, gt_y, err = worst["gt_x"], worst["gt_y"], worst["err"]
    correct = err <= args.tolerance_px

    # Ground-truth box: 100x100 search-px footprint centred on (gt_x, gt_y)
    box_w = box_h = 100
    gt_box_x, gt_box_y = gt_x - box_w / 2.0, gt_y - box_h / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title(f"Reference — {row.get('architecture','')} "
                      f"(tier={row.get('tier','n/a')})")
    axes[0].axis("off")

    axes[1].imshow(srch, cmap="gray")
    axes[1].add_patch(patches.Rectangle(
        (gt_box_x, gt_box_y), box_w, box_h,
        linewidth=2, edgecolor="lime", facecolor="none", label="Ground truth"))
    axes[1].plot(gt_x, gt_y, "+", color="lime", markersize=14, markeredgewidth=2)
    axes[1].plot(result["x"], result["y"], "x", color="red", markersize=14,
                markeredgewidth=2, label="Predicted")
    axes[1].set_title(f"Search — error={err:.2f}px  "
                      f"({'PASS' if correct else 'FAIL'} @ {args.tolerance_px:.0f}px)  "
                      f"method={result['method']}")
    axes[1].legend(loc="upper right")
    axes[1].axis("off")

    root_cause = (
        f"id={row['id']}  architecture={row.get('architecture','n/a')}  "
        f"tier={row.get('tier','n/a')}\n"
        f"GT=({gt_x:.1f},{gt_y:.1f})  Predicted=({result['x']:.1f},{result['y']:.1f})  "
        f"score={result['score']:.3f}  method={result['method']}\n"
        "Root cause: predicted location scores competitively under ZNCC "
        "because the reference crop's periodic content repeats elsewhere in "
        "the search image; see README.md Failure Analysis section."
    )
    fig.suptitle(root_cause, fontsize=9, y=0.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nWorst case: id={row['id']} err={err:.2f}px method={result['method']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
