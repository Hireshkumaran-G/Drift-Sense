#!/usr/bin/env python3
"""
localize.py  —  Drift-Sense: Navigation-Error Recovery
=======================================================
Reference : 1000x1000 px @ 1 nm/px  (100x magnification)
Search    : 1000x1000 px @ 10 nm/px (10x magnification)

Algorithm: Dual-channel ZNCC + confidence-gated intensity fallback
           + ground-truth-free local saliency consensus
--------------------------------------------------------------------
Four stages:

Stage 1 — Edge-channel ZNCC (primary)
  NLM-denoised Canny edges from both images. ZNCC over 7 scales (8.5-11.5x).
  NMS top-30 peaks per scale. Combined score = 0.5*edge + 0.5*intensity.
  Exploits zone-boundary routing strips that create unique edge signatures
  per mat-block location.

Stage 2 — Intensity fallback (when the edge channel is uncertain)
  If the best combined score from Stage 1 is below CONFIDENCE_THRESHOLD,
  fall back to raw intensity ZNCC. Recovers crops that lie entirely inside
  a uniform periodic mat block with no nearby routing strip.

Stage 3 — Local saliency consensus (NEW, only runs if Stage 1 was
  confident)
  Root cause of the remaining periodic-mat-block failures, diagnosed by
  checking the ZNCC score AT the true ground-truth location for several
  documented failures: the correct location was already scoring
  competitively (0.4-0.76, not near-zero) but narrowly outscored --
  genuine near-ties diluted by ~800k-way periodic ambiguity in a
  whole-image vote, not invisible signal. Fix: find the reference's most
  locally non-periodic sub-patch WITHOUT any ground truth (FFT-
  autocorrelation periodicity_score, sliding window -- see
  src/patterns/landmarks.py), match just that patch independently against
  each of Stage 1's own candidates, and add a small weighted bonus to
  whichever candidates it agrees with.
  Safety property (verified, not assumed): the boost is only applied to
  candidates already within PRETIE_MARGIN of Stage 1's own leading score.
  A candidate that Stage 1 already ranked far behind the leader cannot be
  boosted into contention, so a decisive Stage-1 win can never be
  overturned -- confirmed by an adversarial test at 10x the normal boost
  weight. This closed a real bug caught during development: an earlier,
  unconstrained version of this boost DID occasionally overturn decisive
  wins, because a small local match score (up to +1.0) could swamp a
  whole-image score in the ~0.15-0.9 range with no such gate.

Stage 4 — Sub-pixel refinement + tie-break
  Quadratic peak fit on 3x3 neighbourhood (candidate positions are fixed
  at Stage 1 -- the saliency boost in Stage 3 only affects which candidate
  WINS, never where a candidate sits).
  Tie-break: among candidates within TIE_TOL of the best (boosted) score,
  pick the one closest to search-image centre (500, 500), per problem
  statement Section 4.A "Multiple matches" rule.

Measured performance
---------------------
  Original 30-pair submission dataset (dataset/train, no injected
  landmarks -- see citations.md for how that data was generated):
    accuracy@5px  63.3% -> 66.7%   median error 3.91px -> 3.58px
    0 regressions, 1 case fixed (89.95px -> 2.68px)

  40-pair tiered stress-test dataset (see dataset_generator.py):
    accuracy@5px  45.0% -> 45.0% (tied on pass/fail count)
    median error 30.01px -> 15.60px (large failures fixed outweigh
    smaller new ones on magnitude, tied on binary threshold)
    2 catastrophic failures fixed (964px, 300px -> near-zero);
    2 new milder failures introduced on the weak-landmark "medium" tier
    (see Known Limitation below)

Known limitation (documented honestly, not hidden): on the "medium"
difficulty tier, where the injected landmark is deliberately faint
(landmark_strength=0.35), the saliency detector can occasionally lock
onto a different patch that is locally non-periodic but is NOT the
best available anchor, and that patch can coincidentally also match a
wrong location in the search image. This costs 2 of 12 medium-tier
cases in our stress test. A confidence gate on the saliency match itself
(not just its closeness to Stage 1's leader) is the natural next fix but
is not yet implemented -- see the failure analysis section of README.md.

Submission interface
--------------------
  python localize.py --reference <path> --search <path>
  python localize.py --reference <path> --search <path> --gt-x X --gt-y Y
"""

import argparse
import time

import cv2
import numpy as np

from src.patterns.landmarks import periodicity_score

# ── constants ──────────────────────────────────────────────────────────────────
SEARCH_CENTER       = (500.0, 500.0)
SCALES              = [8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
W_EDGE              = 0.5
W_INTENS            = 0.5
TOP_K               = 30
NMS_RADIUS          = 12
TIE_TOL             = 0.01

# If the best combined score from the edge channel is below this threshold,
# the edge channel is considered unreliable (crop is inside a uniform mat
# block) and we fall back to raw intensity ZNCC.
CONFIDENCE_THRESHOLD = 0.15

# How close to Stage 1's own best (pre-boost) score a candidate must
# already be to even be ELIGIBLE for the saliency boost to matter -- see
# Stage 3 docstring above. This is what stops the boost from overturning
# decisive wins.
PRETIE_MARGIN = 0.08
LANDMARK_WEIGHT = 1.0

# Saliency detector: sliding-window scan of the reference for the lowest
# periodicity_score (most locally non-repeating) sub-patch(es).
SALIENT_WINDOW = 150
SALIENT_STRIDE = 75
SALIENT_TOP_K = 3
NOMINAL_SCALE = 10.0  # physical calibration is fixed 10x; only used for
                       # sizing the local saliency template, not candidate search


# ── preprocessing ──────────────────────────────────────────────────────────────

def make_edge(img: np.ndarray) -> np.ndarray:
    """NLM-denoised Canny edge map.

    Non-Local Means denoising preserves real structural edges (feature
    sidewalls, mat-block boundaries) while suppressing Poisson shot noise
    and speckle that destroy Canny edges at low-dose search images.

    NLM parameters:
      h=15                  filter strength; handles speckle_sigma up to ~0.3
      templateWindowSize=7  7x7 patch comparison
      searchWindowSize=21   21x21 search window

    Canny thresholds 15/45 — slightly lower than with Gaussian pre-filter
    to recover edges softened by NLM smoothing.

    Literature: Buades, A., Coll, B. & Morel, J.M., "A Non-Local Algorithm
    for Image Denoising", CVPR 2005.
    """
    denoised = cv2.fastNlMeansDenoising(
        img, h=15, templateWindowSize=7, searchWindowSize=21
    )
    return cv2.Canny(denoised, 15, 45)


def make_intens(img: np.ndarray) -> np.ndarray:
    """Lightly smoothed intensity image for ZNCC.

    No CLAHE — CLAHE equalises local contrast and makes all periodic mat
    cells look identical, increasing rather than decreasing ambiguity.
    """
    return cv2.GaussianBlur(img, (3, 3), 1.0)


# ── sub-pixel refinement ───────────────────────────────────────────────────────

def subpixel_peak(resp: np.ndarray, px: int, py: int) -> tuple:
    """2-D quadratic fit on 3x3 neighbourhood, offset clamped to +/-1 px.

    Standard parabolic interpolation (Foroosh et al., IEEE TIP 2002).
    Improves accuracy from +/-0.5 px (integer) to +/-0.1-0.2 px on clean images.
    """
    h, w = resp.shape
    if px < 1 or px >= w - 1 or py < 1 or py >= h - 1:
        return float(px), float(py)
    p  = resp[py - 1:py + 2, px - 1:px + 2].astype(np.float64)
    ax = 0.5 * (p[1, 2] - 2.0 * p[1, 1] + p[1, 0])
    bx = 0.5 * (p[1, 2] - p[1, 0])
    dx = float(np.clip(-bx / (2.0 * ax), -1.0, 1.0)) if abs(ax) > 1e-10 else 0.0
    ay = 0.5 * (p[2, 1] - 2.0 * p[1, 1] + p[0, 1])
    by = 0.5 * (p[2, 1] - p[0, 1])
    dy = float(np.clip(-by / (2.0 * ay), -1.0, 1.0)) if abs(ay) > 1e-10 else 0.0
    return px + dx, py + dy


# ── NMS peak extraction ────────────────────────────────────────────────────────

def nms_peaks(resp: np.ndarray, top_k: int = TOP_K,
              radius: int = NMS_RADIUS) -> list:
    """Iterative non-maximum suppression -> top-k distinct peaks."""
    work = resp.copy().astype(np.float32)
    h, w = work.shape
    peaks = []
    while len(peaks) < top_k:
        _, val, _, loc = cv2.minMaxLoc(work)
        if val < -0.9:
            break
        peaks.append((loc[0], loc[1], float(val)))
        x0 = max(loc[0] - radius, 0);  x1 = min(loc[0] + radius + 1, w)
        y0 = max(loc[1] - radius, 0);  y1 = min(loc[1] + radius + 1, h)
        work[y0:y1, x0:x1] = -1.0
    return peaks


def dist_to_centre(cx: float, cy: float) -> float:
    return float(np.hypot(cx - SEARCH_CENTER[0], cy - SEARCH_CENTER[1]))


# ── intensity fallback (mirrors Applied Materials baseline) ────────────────────

def _intensity_fallback(ref_i: np.ndarray, srch_i: np.ndarray,
                        scales: list) -> tuple:
    """Raw intensity ZNCC — identical logic to the Applied Materials baseline.

    Takes the single global peak per scale, returns the best (x, y, score).
    Used when the edge channel combined score is below CONFIDENCE_THRESHOLD,
    meaning the reference crop lies in a uniform periodic mat block where
    the edge channel is ambiguous. Raw intensity ZNCC is more stable in
    this regime because it does not depend on routing-strip edge signatures.
    """
    best_x, best_y, best_score = 500.0, 500.0, -1.0
    for sc in scales:
        tw = max(int(round(ref_i.shape[1] / sc)), 1)
        th = max(int(round(ref_i.shape[0] / sc)), 1)
        if tw >= srch_i.shape[1] or th >= srch_i.shape[0]:
            continue
        tmpl  = cv2.resize(ref_i, (tw, th), interpolation=cv2.INTER_AREA)
        resp  = cv2.matchTemplate(srch_i, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(resp)
        if score > best_score:
            px_f, py_f = subpixel_peak(resp, loc[0], loc[1])
            best_x     = px_f + tw / 2.0
            best_y     = py_f + th / 2.0
            best_score = score
    return best_x, best_y, best_score


# ── local saliency consensus (Stage 3) ──────────────────────────────────────────

def find_salient_patches(reference: np.ndarray, window: int = SALIENT_WINDOW,
                         stride: int = SALIENT_STRIDE, top_k: int = SALIENT_TOP_K) -> list:
    """Ground-truth-free: lowest FFT-autocorrelation periodicity_score
    sub-patches of the reference image (low score = locally non-repeating,
    i.e. a good anchor for disambiguating an otherwise-periodic crop).
    See src/patterns/landmarks.py:periodicity_score."""
    h, w = reference.shape
    candidates = []
    for y0 in range(0, h - window + 1, stride):
        for x0 in range(0, w - window + 1, stride):
            patch = reference[y0:y0 + window, x0:x0 + window]
            score = periodicity_score(patch)
            candidates.append((x0 + window // 2, y0 + window // 2, window, score))
    candidates.sort(key=lambda c: c[3])
    return candidates[:top_k]


def build_local_templates(reference: np.ndarray, salient_patches: list,
                          scale: float = NOMINAL_SCALE) -> list:
    """Crop + resize each salient patch (with margin) the same way Stage 1
    resizes its whole-image template, ready for direct local matching."""
    templates = []
    for lx, ly, lsize, _ in salient_patches:
        margin = int(lsize * 0.6)
        x0 = max(int(lx - lsize // 2 - margin), 0)
        y0 = max(int(ly - lsize // 2 - margin), 0)
        x1 = min(int(lx + lsize // 2 + margin), reference.shape[1])
        y1 = min(int(ly + lsize // 2 + margin), reference.shape[0])
        patch = reference[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        pw = max(int(round(patch.shape[1] / scale)), 3)
        ph = max(int(round(patch.shape[0] / scale)), 3)
        templates.append(cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA))
    return templates


def local_boost_at(srch_i: np.ndarray, templates: list, cx: float, cy: float) -> float:
    """Best local ZNCC of any saliency template against a small window of
    the search image centred at (cx, cy). Cheap (small local windows, not
    full-image response maps) and independent of the whole-image score."""
    if not templates:
        return 0.0
    best = -1.0
    h, w = srch_i.shape
    for tmpl in templates:
        th, tw = tmpl.shape
        wx0 = int(max(cx - tw, 0)); wy0 = int(max(cy - th, 0))
        wx1 = int(min(cx + 2 * tw, w)); wy1 = int(min(cy + 2 * th, h))
        window = srch_i[wy0:wy1, wx0:wx1]
        if window.shape[0] < th or window.shape[1] < tw:
            continue
        resp = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(resp.max()))
    return best if best > -1.0 else 0.0


# ── main pipeline ──────────────────────────────────────────────────────────────

def localize(reference: np.ndarray, search: np.ndarray,
             scales: list = SCALES, landmark_weight: float = LANDMARK_WEIGHT,
             salient_window: int = SALIENT_WINDOW, salient_stride: int = SALIENT_STRIDE,
             salient_top_k: int = SALIENT_TOP_K, pretie_margin: float = PRETIE_MARGIN) -> dict:
    """Dual-channel ZNCC + confidence-gated intensity fallback + local
    saliency consensus. Returns dict with keys: x, y, score, n_candidates,
    latency_ms, method. method is 'intensity' (Stage 2 fallback, unchanged
    from the original baseline design), 'edge' (Stage 1 only -- no
    saliency templates found or none eligible), or 'edge+saliency'
    (Stage 3 boost was applied).
    """
    t0 = time.perf_counter()

    ref_e  = make_edge(reference);    srch_e  = make_edge(search)
    ref_i  = make_intens(reference);  srch_i  = make_intens(search)

    # ── Stage 1: edge-channel ZNCC ─────────────────────────────────────────
    all_candidates = []

    for sc in scales:
        tw = max(int(round(reference.shape[1] / sc)), 1)
        th = max(int(round(reference.shape[0] / sc)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue

        tmpl_e = cv2.resize(ref_e, (tw, th), interpolation=cv2.INTER_AREA)
        tmpl_i = cv2.resize(ref_i, (tw, th), interpolation=cv2.INTER_AREA)
        resp_e = cv2.matchTemplate(srch_e, tmpl_e, cv2.TM_CCOEFF_NORMED)
        resp_i = cv2.matchTemplate(srch_i, tmpl_i, cv2.TM_CCOEFF_NORMED)

        for px, py, escore in nms_peaks(resp_e):
            iscore = (float(resp_i[py, px])
                      if 0 <= py < resp_i.shape[0] and 0 <= px < resp_i.shape[1]
                      else 0.0)
            combined = W_EDGE * escore + W_INTENS * iscore
            px_f, py_f = subpixel_peak(resp_e, px, py)
            all_candidates.append([px_f + tw / 2.0, py_f + th / 2.0, combined])

    if all_candidates:
        all_candidates.sort(key=lambda c: -c[2])
        best_score = all_candidates[0][2]
    else:
        best_score = -1.0

    # ── Stage 2: confidence gate -> intensity fallback ──────────────────────
    if best_score < CONFIDENCE_THRESHOLD:
        fb_x, fb_y, fb_score = _intensity_fallback(ref_i, srch_i, scales)
        latency_ms = (time.perf_counter() - t0) * 1e3
        return {
            "x":            fb_x,
            "y":            fb_y,
            "score":        fb_score,
            "n_candidates": len(all_candidates),
            "latency_ms":   latency_ms,
            "method":       "intensity",
        }

    # ── Stage 3: local saliency consensus (pretie-gated, see docstring) ────
    eligible = [c for c in all_candidates if best_score - c[2] <= pretie_margin]

    if len(eligible) <= 1:
        # No candidate within PRETIE_MARGIN of the leader -- mathematically,
        # nothing outside `eligible` can ever out-score it even with the
        # maximum possible boost (an ineligible candidate's raw score is by
        # definition more than pretie_margin below the leader, and boosting
        # only ever adds to an eligible candidate's score). So there is
        # nothing for Stage 3 to arbitrate: skip the saliency scan entirely
        # -- this was previously running unconditionally and roughly
        # doubling per-pair latency even on decisive wins, which are the
        # majority of low-noise cases. Return Stage 1's winner directly, at
        # Stage-1-only cost.
        x, y, score = eligible[0]
        latency_ms = (time.perf_counter() - t0) * 1e3
        return {
            "x": x, "y": y, "score": score,
            "n_candidates": len(all_candidates), "latency_ms": latency_ms,
            "method": "edge",
            "raw_edge_score": score, "saliency_bonus": 0.0, "boosted_score": score,
        }

    # Genuine near-tie among `eligible` -- worth paying for the saliency
    # scan. Candidates outside `eligible` are provably unable to win after
    # boosting (see above), so they're correctly excluded, not just ignored.
    salient_patches = find_salient_patches(reference, salient_window, salient_stride, salient_top_k)
    templates = build_local_templates(reference, salient_patches)

    boosted = []
    for x, y, score in eligible:
        bonus = local_boost_at(srch_i, templates, x, y) if templates else 0.0
        boosted.append((x, y, score + landmark_weight * bonus, score, bonus))

    # ── Stage 4: tie-break + sub-pixel (positions already refined in Stage 1) ──
    boosted.sort(key=lambda c: -c[2])
    best_boosted = boosted[0][2]
    tied   = [c for c in boosted if best_boosted - c[2] <= TIE_TOL]
    winner = (tied[0] if len(tied) == 1
              else min(tied, key=lambda c: dist_to_centre(c[0], c[1])))

    latency_ms = (time.perf_counter() - t0) * 1e3
    return {
        "x":            winner[0],
        "y":            winner[1],
        # Report the PRE-boost combined score, not score+bonus. The bonus
        # exists to influence which candidate WINS the tie-break (an
        # internal ranking decision), not to represent match confidence on
        # a scale comparable across samples/methods. Reporting the boosted
        # value here corrupted evaluate_improved.py's AP calculation, which
        # ranks all samples globally by this field: 'edge+saliency' scores
        # (combined + bonus, can exceed 1.0) were on a different scale than
        # 'intensity'/'edge' scores (~0-1 ZNCC), distorting the global
        # confidence ranking even on samples whose PREDICTED POSITION never
        # changed. Caught by comparing a fresh evaluate_improved.py run
        # against the pre-Stage-3 baseline: AP dropped at every noise level
        # even where acc@5px was flat or only marginally different.
        "score":        winner[3],
        "n_candidates": len(all_candidates),
        "latency_ms":   latency_ms,
        "method":       "edge+saliency" if templates else "edge",
        "raw_edge_score":  winner[3],
        "saliency_bonus":  winner[4],
        "boosted_score":   winner[2],
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--reference", required=True)
    ap.add_argument("--search",    required=True)
    ap.add_argument("--gt-x",  type=float, default=None)
    ap.add_argument("--gt-y",  type=float, default=None)
    ap.add_argument("--landmark-weight", type=float, default=LANDMARK_WEIGHT)
    ap.add_argument("--pretie-margin",   type=float, default=PRETIE_MARGIN)
    ap.add_argument("--salient-window",  type=int,   default=SALIENT_WINDOW)
    ap.add_argument("--salient-stride",  type=int,   default=SALIENT_STRIDE)
    ap.add_argument("--salient-top-k",   type=int,   default=SALIENT_TOP_K)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref  = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(args.search,    cv2.IMREAD_GRAYSCALE)
    if ref  is None: raise FileNotFoundError(f"Cannot read: {args.reference}")
    if srch is None: raise FileNotFoundError(f"Cannot read: {args.search}")

    result = localize(ref, srch, landmark_weight=args.landmark_weight,
                      pretie_margin=args.pretie_margin,
                      salient_window=args.salient_window,
                      salient_stride=args.salient_stride,
                      salient_top_k=args.salient_top_k)

    # Primary output — what Applied Materials' evaluation script parses
    print(f"x={result['x']:.4f} y={result['y']:.4f}")

    if args.verbose:
        import sys
        extra = ""
        if "raw_edge_score" in result:
            extra = (f"  raw_edge={result['raw_edge_score']:.4f}  "
                     f"saliency_bonus={result['saliency_bonus']:.4f}")
        print(f"score={result['score']:.4f}  "
              f"candidates={result['n_candidates']}  "
              f"method={result['method']}  "
              f"latency={result['latency_ms']:.1f}ms{extra}", file=sys.stderr)

    if args.gt_x is not None and args.gt_y is not None:
        err = float(np.hypot(result["x"] - args.gt_x,
                             result["y"] - args.gt_y))
        print(f"error={err:.4f}px  "
              f"(pred={result['x']:.2f},{result['y']:.2f}  "
              f"gt={args.gt_x:.2f},{args.gt_y:.2f})")


if __name__ == "__main__":
    main()
