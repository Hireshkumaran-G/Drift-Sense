"""
Pattern-collapse / bridging and missing contact defects.

Both model real *structural* defects — properties of the physical device,
not of how it was imaged.  Applied once to the fine canvas so they appear
consistently in both reference crop and derived search image.

Changes vs starter
------------------
+ add_missing_contacts() : randomly voids a fraction of contact/via circles,
  simulating etch failure, blocked CMP, or resist collapse.
  Literature: ITRS/IRDS yield models; Nishi & Doering, "Handbook of
  Semiconductor Manufacturing Technology", 2nd ed., CRC 2007 — Chapter 9.
"""

import numpy as np


def maybe_collapse_gap(
    gap_nm: float,
    threshold_nm: float,
    rng: np.random.Generator,
    collapse_prob: float = 0.7,
) -> bool:
    """Return True if a gap between adjacent lines should bridge/merge.

    Gaps at or above threshold_nm never collapse.  Below threshold they
    collapse with probability collapse_prob — mirrors real process variation.
    """
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)


def add_missing_contacts(
    contact_list: list,
    rng: np.random.Generator,
    missing_fraction: float = 0.07,
) -> list:
    """Randomly remove a fraction of contacts/vias from a contact list.

    Parameters
    ----------
    contact_list : list
        Each element is a dict or tuple describing one contact (as produced
        by src/patterns/dram.py).  The format is passed through unchanged
        for surviving contacts.
    rng : np.random.Generator
    missing_fraction : float
        Fraction of contacts to void (default 7 %).
        Typical values in literature: 5–15 % for demonstration purposes.
        Set to 0 to disable.

    Returns
    -------
    list : surviving contacts (order preserved, missing ones dropped).
    """
    if missing_fraction <= 0 or not contact_list:
        return contact_list
    mask = rng.random(len(contact_list)) >= missing_fraction
    return [c for c, keep in zip(contact_list, mask) if keep]


def apply_linewidth_roughness(
    width_nm: float,
    rng: np.random.Generator,
    lwr_sigma_nm: float = 1.5,
) -> float:
    """Return a perturbed line width (LWR — line-width roughness).

    LWR is a well-characterised stochastic effect in EUV and ArF lithography.
    Here we model it as Gaussian per-line variation with sigma ≈ 1.5 nm,
    consistent with reported 3σ LWR of 4–6 nm for sub-20-nm pitches
    (Mack, "Fundamental Principles of Optical Lithography", Wiley 2007).
    """
    return max(width_nm + float(rng.normal(0, lwr_sigma_nm)), 1.0)
