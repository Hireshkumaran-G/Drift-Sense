# Drift-Sense: Literature Citations

All synthetic data generation choices — structure dimensions, noise models,
augmentation types, and algorithmic decisions — are justified below using
credible public sources. Citations correspond to references in the solution
presentation and README.

The principle followed throughout: citations justify the physical phenomenon
or established architecture — not the exact numerical parameter or
mathematical implementation chosen. Simulator parameters are engineering
choices informed by, but not directly mandated by, the cited literature.

---

## 1. Semiconductor Structure Dimensions

**[1] FreePDK15 Predictive Process Design Kit**
- Bhanushali, K., Tembe, C. & Davis, W.R., NC State University / Arizona State University
- "Development of a Predictive Process Design Kit for 15-nm FinFETs: FreePDK15"
- arXiv:2009.04600, 2020
- Establishes publicly available 15 nm-node FinFET design rules including
  fin pitch, gate contacted-poly-pitch (CPP), and gate length values.
- Used to justify: `finfet_10nm`, `finfet_7nm` presets in `src/presets.py`.
  Specific pitch values in presets are illustrative and consistent with the
  scaling trends described in this PDK; they are not claimed as exact
  proprietary fabrication specifications.

**[2] US Patent US7349232B2 — 6F² DRAM Cell Design**
- United States Patent, granted 2008
- Describes 6F² DRAM cell architecture with 3F-pitch folded digitline sense
  amplifier, word lines, active areas, and storage-node layout in a
  90° orthogonal (Manhattan geometry) array.
- Used to justify: all DRAM presets in `src/presets.py` — specifically the
  6F² cell geometry (word-line pitch = 2F, bit-line pitch = 3F) and
  orthogonal Manhattan layout constraint.

**[3] Imec 18 nm Metal Pitch Semi-Damascene Interconnects**
- Marti, G. et al., Imec
- "Two-Level Semi-Damascene Interconnect with Fully Self-Aligned Vias at MP18"
- IITC/MAM 2023, DOI: 10.1109/IITC/MAM57687.2023.10154682
- Describes advanced BEOL interconnect at 18 nm metal pitch, providing
  reference dimensions for metal line and via scaling at advanced nodes.
- Used to justify: contact and via scaling trends in DRAM and FinFET presets.
  Specific contact dimensions in presets are synthetic approximations
  consistent with published scaling trends, not exact fab specifications.

**[4] IRDS — Yield Enhancement Chapter, 2022**
- IEEE International Roadmap for Devices and Systems, 2022
- Available: https://irds.ieee.org/
- Discusses semiconductor defect occurrence, defect density targets, CD
  variation, and yield modelling relevant to synthetic defect simulation.
- Used to motivate: structural defect simulation in `src/structural_defects.py`
  (missing contacts, gap collapse). The 7% missing-contact probability is a
  synthetic stress parameter chosen for dataset diversity; it is not claimed
  as an industry-standard defect rate.

---

## 2. SEM Imaging Physics

**[5] Goldstein, J.I., Newbury, D.E., Michael, J.R., Ritchie, N.W.M., Scott, J.H.J. & Joy, D.C.**
- *Scanning Electron Microscopy and X-ray Microanalysis*, 4th edition
- Springer, 2018. ISBN 978-1-4939-6674-5
- Chapter 3: Secondary electron yield increases near sharp edges and inclined
  sidewalls due to enhanced escape probability — physical basis for
  edge-brightening in SEM images of semiconductor structures.
- Chapter 4: Electron detection involves Poisson-distributed counting
  statistics, motivating shot-noise models for SEM image simulation.
- Chapter 14: Surface charging induces local intensity distortions and
  image artifacts in low-conductivity specimens.
- Used to motivate: `add_edge_brightening()`, `add_shot_noise()`,
  `add_charging_streaks()`, `gaussian_psf_blur()` in `src/sem_imaging.py`.
  Mathematical implementations are synthetic approximations consistent
  with the physical phenomena described.

**[6] Reimer, L.**
- *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*
- 2nd edition, Springer, 1998. ISBN 978-3-540-63976-3
- Detector noise (additive read noise from scintillator/PMT chain) motivates
  Gaussian detector noise model.
- Raster-scan beam positional errors accumulate across the scan frame,
  motivating row-by-row shear and jitter simulation.
- Lens aberrations including astigmatism motivate elliptical Gaussian PSF.
- Off-axis collection efficiency falloff motivates vignetting simulation.
- Used to motivate: `add_detector_noise()`, `apply_raster_drift()`,
  `gaussian_psf_blur()` (astigmatism), `apply_vignette()` in `src/sem_imaging.py`.

---

## 3. Industrial Alignment Reference

**[7] Applied Materials / KLA — Semiconductor Wafer Metrology and Alignment**
- Brill, B. et al., KLA-Tencor Technologies Corporation
- US Patent US7561282B2 — "Techniques for determining overlay and critical
  dimension using a single metrology tool"
- Provides industrial context for multi-scale image registration and
  pattern alignment in semiconductor wafer inspection tools.
- Used to motivate: the overall cross-magnification (100× → 10×) localization
  problem formulation and the need for robust navigation-error recovery in
  high-volume manufacturing inspection.

---

## 4. Structural Defects

**[8] Nishi, Y. & Doering, R. (eds.)**
- *Handbook of Semiconductor Manufacturing Technology*, 2nd edition
- CRC Press, 2007. ISBN 978-1-57444-675-3
- Discusses semiconductor yield and defectivity, including open-circuit
  failures from missing contacts and short-circuit failures from
  insufficient inter-feature spacing (line bridging).
- Used to motivate: `add_missing_contacts()` and `maybe_collapse_gap()`
  in `src/structural_defects.py` as physically plausible structural defects.
  Specific defect probabilities are synthetic parameters, not claimed
  as values from this reference.

**[9] Mack, C.**
- *Fundamental Principles of Optical Lithography*
- Wiley-Blackwell, 2007. ISBN 978-0-470-01893-4
- Discusses line-width roughness (LWR) as a stochastic variation in
  printed line width arising from photon/electron shot noise and
  resist chemistry, relevant to sub-20 nm pitch processes.
- Used to motivate: `apply_linewidth_roughness()` in `src/structural_defects.py`.
  The σ=1.5 nm LWR parameter is a synthetic choice consistent with the
  scale of LWR effects discussed in lithography literature; it is not
  claimed as a specific value from this reference.

---

## 5. Localization Algorithm

**[10] Buades, A., Coll, B. & Morel, J.M.**
- "A Non-Local Algorithm for Image Denoising"
- Proceedings of IEEE CVPR, Vol. 2, pp. 60–65, 2005
- DOI: 10.1109/CVPR.2005.38
- Introduces Non-Local Means (NLM) denoising, which performs non-local
  weighted averaging to preserve image structures more effectively than
  conventional local smoothing filters.
- Used to justify: `cv2.fastNlMeansDenoising()` pre-processing in `localize.py`.
  NLM was selected because it suppresses Poisson shot noise and speckle
  while preserving feature edges, which is critical for the subsequent
  Canny edge extraction step.

**[11] Foroosh, H., Zerubia, J.B. & Berthod, M.**
- "Extension of Phase Correlation to Subpixel Registration"
- IEEE Transactions on Image Processing, 11(3), pp. 188–200, 2002
- DOI: 10.1109/83.988953
- Derives subpixel translation estimation from correlation peak neighbourhoods.
- Used to motivate: `subpixel_peak()` in `localize.py`, which applies
  quadratic interpolation on a 3×3 neighbourhood of the correlation peak
  to refine the integer-pixel localization to sub-pixel accuracy.

**[12] Lewis, J.P.**
- "Fast Normalized Cross-Correlation"
- Vision Interface, 1995
- Available: http://scribblethink.org/Work/nvisionInterface/nip.pdf
- Describes normalized cross-correlation for template matching. Normalized
  cross-correlation reduces sensitivity to multiplicative and additive
  intensity differences between template and search image, making it
  suitable for cross-magnification matching where reference and search
  images are acquired under different imaging conditions.
- Used to justify: `cv2.matchTemplate(TM_CCOEFF_NORMED)` as the primary
  similarity metric in `localize.py`.

---

## 6. Local Saliency / Periodicity Detection

**[13] Bracewell, R.N.**
- *The Fourier Transform and Its Applications*, 3rd edition
- McGraw-Hill, 2000. ISBN 978-0-07-303938-1
- The Wiener–Khinchin theorem: the autocorrelation of a signal equals the
  inverse Fourier transform of its power spectral density. This is the
  exact identity `periodicity_score()` in `src/patterns/landmarks.py`
  uses to compute a crop's autocorrelation via `ifft2(F * conj(F))`
  rather than a direct (much slower) spatial-domain autocorrelation —
  a standard, textbook technique, not a novel contribution.
- Used to justify: `periodicity_score()` and the sliding-window saliency
  scan in `localize.py`'s Stage 3 (`find_salient_patches`), which locates
  the reference crop's most locally non-periodic sub-patch without any
  ground truth, to disambiguate matches in an otherwise highly periodic
  DRAM/FinFET array. The *decision* to use a secondary-autocorrelation-
  peak ratio as an ambiguity measure, and to use it as a local matching
  cue on top of (not instead of) the whole-image ZNCC score, is this
  project's own design choice, arrived at empirically (see the failure
  analysis in README.md) — the citation covers the underlying signal-
  processing identity, not that specific application.
