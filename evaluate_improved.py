#!/usr/bin/env python3
"""
evaluate_improved.py  —  Drift-Sense Localization Evaluation
=============================================================
Runs the improved ZNCC localizer (localize.py) across noise levels,
architecture types, and boundary vs interior crops.

Produces
--------
  eval_results/pr_curves.png
  eval_results/ap_vs_noise.png
  eval_results/metrics_table.csv
  eval_results/summary.txt

Usage
-----
  python evaluate_improved.py --samples-per-level 40 --tolerance-px 5
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import GenerationParams, generate_sample
from localize import localize

# numpy ≥2.0 renamed trapz → trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

NOISE_LEVELS = [
    {
        "label": "low",
        "dose_search": 800.0,
        "detector_noise_sigma_search": 2.0,
        "shear_amplitude_px": 0.5,
        "drift_jitter_px": 0.2,
    },
    {
        "label": "medium",
        "dose_search": 200.0,
        "detector_noise_sigma_search": 5.0,
        "shear_amplitude_px": 1.5,
        "drift_jitter_px": 0.5,
    },
    {
        "label": "high",
        "dose_search": 60.0,
        "detector_noise_sigma_search": 8.0,
        "shear_amplitude_px": 2.5,
        "drift_jitter_px": 1.0,
        "speckle_sigma": 0.15,
    },
    {
        "label": "severe",
        "dose_search": 25.0,
        "detector_noise_sigma_search": 12.0,
        "shear_amplitude_px": 4.0,
        "drift_jitter_px": 1.8,
        "speckle_sigma": 0.3,
        "salt_pepper_prob": 0.01,
    },
]

ARCHITECTURES = ["dram_1x", "dram_dense", "finfet_10nm", "finfet_7nm"]
TOLERANCES    = [1.0, 3.0, 5.0]


def pr_curve(scores, corrects, n_total):
    order = np.argsort(-np.asarray(scores))
    c  = np.asarray(corrects)[order]
    tp = np.cumsum(c)
    fp = np.cumsum(~c)
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / max(n_total, 1)
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])
    return prec, rec


def average_precision(prec, rec):
    order = np.argsort(rec)
    return float(_trapz(prec[order], rec[order]))


def evaluate_level(level, n_samples, tolerance_px, base_seed):
    rng      = np.random.default_rng(base_seed)
    overrides = {k: v for k, v in level.items() if k != "label"}

    scores, errors, latencies            = [], [], []
    boundary_correct, interior_correct   = [], []

    for i in range(n_samples):
        arch   = ARCHITECTURES[i % len(ARCHITECTURES)]
        params = GenerationParams(**overrides)
        sample = generate_sample(arch, rng, params)

        ref  = sample["reference_img"]
        srch = sample["search_img"]

        t0     = time.perf_counter()
        result = localize(ref, srch)
        lat    = (time.perf_counter() - t0) * 1e3

        err     = float(np.hypot(result["x"] - sample["gt_x"],
                                 result["y"] - sample["gt_y"]))
        correct = err <= tolerance_px

        scores.append(result["score"])
        errors.append(err)
        latencies.append(lat)

        # boundary vs interior
        is_boundary = False
        gt_x0, gt_y0, gw, gh = sample["gt_box"]
        for (sx, sy, sw, sh) in sample.get("strip_rects", []):
            if (gt_x0 < sx + sw and gt_x0 + gw > sx and
                    gt_y0 < sy + sh and gt_y0 + gh > sy):
                is_boundary = True
                break
        if is_boundary:
            boundary_correct.append(correct)
        else:
            interior_correct.append(correct)

        # live progress
        marker = "✓" if correct else "✗"
        print(f"  [{i+1:3d}/{n_samples}] {arch:15s} "
              f"err={err:6.2f}px {marker}  "
              f"lat={lat:.0f}ms", flush=True)

    errors   = np.array(errors)
    corrects = np.array([e <= tolerance_px for e in errors], dtype=bool)
    prec, rec = pr_curve(scores, corrects, n_samples)
    ap        = average_precision(prec, rec)

    success = {f"success@{int(t)}px": float((errors <= t).mean())
               for t in TOLERANCES}

    return {
        "label":           level["label"],
        "precision":       prec,
        "recall":          rec,
        "ap":              ap,
        "accuracy":        float(corrects.mean()),
        "rmse":            float(np.sqrt((errors ** 2).mean())),
        "median_err":      float(np.median(errors)),
        "mean_latency_ms": float(np.mean(latencies)),
        "n":               n_samples,
        "boundary_acc":    float(np.mean(boundary_correct)) if boundary_correct else float("nan"),
        "interior_acc":    float(np.mean(interior_correct)) if interior_correct else float("nan"),
        **success,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--samples-per-level", type=int,   default=40)
    ap.add_argument("--tolerance-px",      type=float, default=5.0)
    ap.add_argument("--seed",              type=int,   default=123)
    ap.add_argument("--output-dir",        default="./eval_results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for i, level in enumerate(NOISE_LEVELS):
        print(f"\n── Noise level: {level['label']} "
              f"({args.samples_per_level} samples) ──")
        res = evaluate_level(
            level, args.samples_per_level,
            args.tolerance_px, args.seed + i * 9973,
        )
        results.append(res)
        print(f"\n  AP={res['ap']:.3f}  "
              f"acc@{args.tolerance_px:.0f}px={res['accuracy']:.3f}  "
              f"RMSE={res['rmse']:.2f}px  "
              f"median={res['median_err']:.2f}px  "
              f"latency={res['mean_latency_ms']:.1f}ms")
        print(f"  @1px={res['success@1px']:.3f}  "
              f"@3px={res['success@3px']:.3f}  "
              f"@5px={res['success@5px']:.3f}")
        print(f"  boundary={res['boundary_acc']:.3f}  "
              f"interior={res['interior_acc']:.3f}")

    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        ax.plot(r["recall"], r["precision"], marker="o", markersize=3,
                label=f"{r['label']} (AP={r['ap']:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Drift-Sense: PR curves by noise level "
                 f"(tol={args.tolerance_px}px)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(); ax.grid(alpha=0.3)
    pr_path = os.path.join(args.output_dir, "pr_curves.png")
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # AP + RMSE vs noise
    labels = [r["label"]    for r in results]
    aps    = [r["ap"]       for r in results]
    accs   = [r["accuracy"] for r in results]
    rmses  = [r["rmse"]     for r in results]
    x = np.arange(len(labels))

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(x, aps,  marker="o", label="Average Precision")
    ax2.plot(x, accs, marker="s", label=f"Accuracy (<={args.tolerance_px}px)")
    ax2r = ax2.twinx()
    ax2r.plot(x, rmses, marker="^", color="tab:red", linestyle="--",
              label="RMSE (px)")
    ax2r.set_ylabel("RMSE (px)", color="tab:red")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("Score")
    ax2.set_title("Localizer quality vs noise level")
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, loc="lower left")
    ax2.grid(alpha=0.3)
    trend_path = os.path.join(args.output_dir, "ap_vs_noise.png")
    fig2.savefig(trend_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # metrics CSV
    csv_path = os.path.join(args.output_dir, "metrics_table.csv")
    csv_fields = [
        "label", "ap", "accuracy", "rmse", "median_err",
        "success@1px", "success@3px", "success@5px",
        "mean_latency_ms", "boundary_acc", "interior_acc", "n",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                        for k in csv_fields})

    # summary text
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Drift-Sense Localizer Evaluation Summary\n")
        f.write("=" * 42 + "\n\n")
        for r in results:
            f.write(f"Noise level : {r['label']}\n")
            f.write(f"  Samples   : {r['n']}\n")
            f.write(f"  AP        : {r['ap']:.4f}\n")
            f.write(f"  Accuracy  : {r['accuracy']:.4f}\n")
            f.write(f"  RMSE      : {r['rmse']:.4f} px\n")
            f.write(f"  Median err: {r['median_err']:.4f} px\n")
            f.write(f"  @1px/@3px/@5px : "
                    f"{r['success@1px']:.4f} / "
                    f"{r['success@3px']:.4f} / "
                    f"{r['success@5px']:.4f}\n")
            f.write(f"  Latency   : {r['mean_latency_ms']:.1f} ms/pair\n")
            f.write(f"  Boundary  : {r['boundary_acc']:.4f}  "
                    f"Interior: {r['interior_acc']:.4f}\n\n")

    print(f"\nSaved: {pr_path}")
    print(f"Saved: {trend_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
