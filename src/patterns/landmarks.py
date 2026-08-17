"""
Controlled landmark injection for Drift-Sense Generator V2.
=============================================================
The V1 failure mode (documented in README/eval_log) is periodic
mat-block ambiguity: a crop taken from deep inside a uniform periodic
array has many equally-good matches elsewhere in the search image.

V1's only defense against this was the zone/strip boundary system
(routing strips between mat blocks) combined with `boundary_bias`,
which is binary: a crop either straddles a strip or it doesn't.

V2 makes the difficulty a controllable, continuous, *labelled*
variable by injecting sparse, asymmetric "landmark" glyphs directly
into mat interiors. A landmark is a small non-periodic shape (an L,
a triangle-of-dots, or a notch) that a periodic autocorrelation
cannot reproduce at any other translation -- unlike the routing
strips, it can appear far from any mat/strip boundary, so "landmark
present" and "boundary present" are independent difficulty axes.

Real-world justification: real dies contain non-periodic content
inside otherwise periodic arrays -- redundancy fuses, laser-repair
marks, alignment/verniers, ECC/spare rows, test-structure insertions
(see IRDS Yield Enhancement Chapter 2022, and general foundry design-
for-test literature on embedded redundancy/repair structures). We do
not claim any specific proprietary layout; the glyph shapes here are
synthetic stand-ins for "locally unique, non-repeating content
embedded in a periodic array."
"""

from __future__ import annotations

import cv2
import numpy as np

LANDMARK_VAL = 235
LANDMARK_KINDS = ("L", "triangle", "notch", "cross_offset")


def _draw_glyph(canvas: np.ndarray, cx: int, cy: int, size_px: int,
                 kind: str, value: int) -> None:
    """Stamp one asymmetric glyph centered at (cx, cy)."""
    s = max(size_px, 6)
    h, w = canvas.shape

    def clip(x, y):
        return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    if kind == "L":
        p0 = clip(cx - s // 2, cy - s // 2)
        p1 = clip(cx - s // 2 + max(s // 4, 2), cy + s // 2)
        cv2.rectangle(canvas, p0, p1, value, -1)
        p2 = clip(cx - s // 2, cy + s // 2 - max(s // 4, 2))
        p3 = clip(cx + s // 2, cy + s // 2)
        cv2.rectangle(canvas, p2, p3, value, -1)
    elif kind == "triangle":
        pts = np.array([
            clip(cx, cy - s // 2),
            clip(cx - s // 2, cy + s // 2),
            clip(cx + s // 2, cy + s // 3),
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], value)
    elif kind == "notch":
        p0 = clip(cx - s // 2, cy - s // 2)
        p1 = clip(cx + s // 2, cy + s // 2)
        cv2.rectangle(canvas, p0, p1, value, -1)
        p2 = clip(cx - s // 4, cy - s // 4)
        p3 = clip(cx + s // 4, cy + s // 4)
        cv2.rectangle(canvas, p2, p3, LANDMARK_VAL // 3, -1)
    else:  # cross_offset -- asymmetric cross (arms unequal length)
        cv2.line(canvas, clip(cx - s // 2, cy), clip(cx + s // 4, cy), value, max(s // 6, 2))
        cv2.line(canvas, clip(cx, cy - s // 3), clip(cx, cy + s // 2), value, max(s // 6, 2))


def inject_landmarks(
    canvas: np.ndarray,
    mat_rects: list,
    rng: np.random.Generator,
    count: int,
    strength: float,
    glyph_size_px: int = 26,
) -> list:
    """Stamp `count` asymmetric landmark glyphs into random mat interiors.

    Parameters
    ----------
    strength : float in [0, 1]
        Visual contrast of the landmark relative to background/pattern.
        0 -> invisible (no-op), 1 -> full LANDMARK_VAL brightness.
    Returns
    -------
    list of (x, y, size) landmark centres/sizes in canvas px, for
    manifest bookkeeping / periodicity-label ground truth.
    """
    if count <= 0 or strength <= 0 or not mat_rects:
        return []

    value = int(np.clip(strength, 0.0, 1.0) * LANDMARK_VAL)
    placed = []
    for _ in range(count):
        rx, ry, rw, rh = mat_rects[int(rng.integers(0, len(mat_rects)))]
        margin = glyph_size_px
        if rw <= 2 * margin or rh <= 2 * margin:
            continue
        cx = int(rng.integers(rx + margin, rx + rw - margin))
        cy = int(rng.integers(ry + margin, ry + rh - margin))
        kind = LANDMARK_KINDS[int(rng.integers(0, len(LANDMARK_KINDS)))]
        _draw_glyph(canvas, cx, cy, glyph_size_px, kind, value)
        placed.append((cx, cy, glyph_size_px))
    return placed


def periodicity_score(img: np.ndarray) -> float:
    """Estimate how ambiguous/periodic a crop is via FFT autocorrelation.

    Returns the ratio (secondary peak height) / (zero-shift peak height)
    of the normalized autocorrelation, excluding a small window around
    the zero-shift peak itself. ~0 -> unique/aperiodic content (safe).
    ~1 -> strongly periodic content, many equally good translations
    (the exact ambiguity mode documented as V1's primary failure).
    """
    f = img.astype(np.float64)
    f = f - f.mean()
    norm = np.sqrt((f ** 2).sum())
    if norm < 1e-6:
        return 1.0  # blank/uniform crop is maximally ambiguous
    F = np.fft.fft2(f)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    ac /= (norm ** 2)

    h, w = ac.shape
    cy, cx = h // 2, w // 2
    zero_peak = ac[cy, cx]

    # Mask out a region around the zero-shift peak (~5% of dimension)
    r = max(int(0.05 * min(h, w)), 4)
    masked = ac.copy()
    masked[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] = -np.inf
    secondary = float(masked.max())
    if zero_peak <= 1e-9:
        return 1.0
    return float(np.clip(secondary / zero_peak, 0.0, 1.0))
