"""Feasibility Monte Carlo: azimuth/elevation of a common-source spoof group from
carrier-phase double differences on two (or more) short baselines.

Companion to RESEARCH_next_step_source_azimuth_elevation.md (section 4). Run with
``uv run python simulate_multibaseline_direction.py``; the tables it prints are stored in
``outputs/direction_feasibility/sim_results.txt``. No dataset is read.

Model (per baseline j with vector b_j in ENU, length k_j wavelengths, unit direction bhat_j):
    SD_j(i)  = k_j * bhat_j . u_i + clock_j + N_ij               [cycles]
    DD_j(i)  = SD_j(i) - SD_j(r) = k_j * bhat_j . (u_i - u_r) + integer
    f_j(i)   = wrap(DD_j(i)) + noise                              (centered wrapping, Eq. 18)
    c_j      = bhat_j . u_i  (direction cosine of signal i along baseline j)
             = (f_j + m_j) / k_j + bhat_j . u_r                   (Eq. 21 generalized)
Candidates for c_j: every value congruent to the true c_j modulo 1/k_j inside [-1, 1].
For a spoof group, every spoofed PRN shares u_s, so c_j is the same for all of them.

Given candidate (c_1, c_2) on two non-collinear baselines, the in-plane part of u is fixed
and the out-of-plane part is +-sqrt(1 - |u_par|^2); for a horizontal array only the
upper solution is physical. This script counts how many physical candidates survive, and
propagates the measured fractional-DD noise (sigma_f ~ 0.012 cyc per PRN per epoch in
sp-partial-28-09) into azimuth / elevation errors.
"""
from __future__ import annotations

import itertools
import numpy as np

LAMBDA = 0.19029  # m, GPS L1
rng = np.random.default_rng(1)


def unit(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])  # E, N, U


def az_el(u):
    e, n, up = u
    return (np.rad2deg(np.arctan2(e, n)) % 360.0, np.rad2deg(np.arcsin(np.clip(up, -1, 1))))


def candidates_1d(c_true, k):
    """All direction cosines congruent to c_true modulo 1/k within [-1, 1]."""
    step = 1.0 / k
    out = []
    m = -int(np.ceil(2 * k)) - 1
    while m <= int(np.ceil(2 * k)) + 1:
        c = c_true + m * step
        if -1.0 <= c <= 1.0:
            out.append(c)
        m += 1
    return out


def solve_direction(B, c, require_upper=True, tol=1e-6):
    """B: (nb, 3) unit baseline vectors (rows), c: (nb,) direction cosines.
    Returns list of unit vectors u with B u = c and |u| = 1. With redundant baselines
    (rank < nb) the least-squares residual must be <= tol per equation; with rank 3 the
    norm must be within tol of 1. tol ~ 3 sigma_c lets noise-level inconsistencies pass."""
    U, s, Vt = np.linalg.svd(B, full_matrices=True)
    rank = int((s > 1e-9).sum())
    u_par = np.linalg.pinv(B) @ c
    if B.shape[0] > rank and np.max(np.abs(B @ u_par - c)) > tol:
        return []  # integer combination inconsistent across redundant baselines
    if rank >= 3:
        n2 = u_par @ u_par
        if abs(np.sqrt(n2) - 1.0) > tol:  # inconsistent with a unit vector
            return []
        return [u_par / np.linalg.norm(u_par)]
    n2 = u_par @ u_par
    if n2 > 1.0 + 1e-9:
        return []
    normal = Vt[-1]  # unit vector normal to the baseline plane (rank 2)
    h = np.sqrt(max(0.0, 1.0 - n2))
    sols = [u_par + h * normal, u_par - h * normal]
    if require_upper:
        sols = [u for u in sols if u[2] >= -1e-9]
    return sols


def array_candidates(B, k, u_true, require_upper=True, tol=0.04):
    """Enumerate all physical direction candidates for the true direction u_true.
    tol (cycles-equivalent in direction cosine) is the consistency tolerance for
    redundant baselines; 0.04 ~ 3 sigma of the measured 0.012 cyc fractional-DD noise."""
    c_true = B @ u_true
    per_baseline = [candidates_1d(c_true[j], k[j]) for j in range(len(k))]
    sols = []
    for combo in itertools.product(*per_baseline):
        for u in solve_direction(B, np.array(combo), require_upper, tol=tol):
            sols.append(u)
    # merge duplicates
    merged = []
    for u in sols:
        if not any(np.linalg.norm(u - v) < 1e-6 for v in merged):
            merged.append(u)
    return merged


GEOMETRIES = {
    # name: (unit baseline vectors in ENU, lengths in wavelengths)
    "A. one baseline, D=λ (today)": (np.array([[0, 1, 0.0]]), [1.0]),
    "B. L-shape, both D=λ": (np.array([[0, 1, 0.0], [1, 0, 0.0]]), [1.0, 1.0]),
    "C. equilateral triangle, side λ": (
        np.array([[0, 1, 0.0], [np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60)), 0.0]]),
        [1.0, 1.0],
    ),
    "D. L-shape, D=λ and D=λ/2": (np.array([[0, 1, 0.0], [1, 0, 0.0]]), [1.0, 0.5]),
    "E. L-shape, both D=λ/2": (np.array([[0, 1, 0.0], [1, 0, 0.0]]), [0.5, 0.5]),
    "F. L-shape, both D=0.75λ": (np.array([[0, 1, 0.0], [1, 0, 0.0]]), [0.75, 0.75]),
    "G. L-shape λ/2 + colinear λ (4 antennas, N and E)": (
        np.array([[0, 1, 0.0], [1, 0, 0.0], [0, 1, 0.0], [1, 0, 0.0]]),
        [0.5, 0.5, 1.0, 1.0],
    ),
    "H. L-shape λ (horizontal) + vertical λ/2 (3-D, 4 antennas)": (
        np.array([[0, 1, 0.0], [1, 0, 0.0], [0, 0, 1.0]]),
        [1.0, 1.0, 0.5],
    ),
    "I. two headings of one D=λ baseline (N, then E), synthetic": (
        np.array([[0, 1, 0.0], [1, 0, 0.0]]),
        [1.0, 1.0],
    ),
    "J. three headings of one D=λ baseline (0°, 60°, 120°), synthetic": (
        np.array([[0, 1, 0.0], [np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60)), 0.0],
                  [np.sin(np.deg2rad(120)), np.cos(np.deg2rad(120)), 0.0]]),
        [1.0, 1.0, 1.0],
    ),
    "K. square side λ (N, E, diagonal √2λ; 4 antennas)": (
        np.array([[0, 1, 0.0], [1, 0, 0.0], [np.sqrt(0.5), np.sqrt(0.5), 0.0]]),
        [1.0, 1.0, np.sqrt(2.0)],
    ),
    "L. vertical L: N λ/2 + Up λ/2 (3 antennas, plane N-U)": (
        np.array([[0, 1, 0.0], [0, 0, 1.0]]),
        [0.5, 0.5],
    ),
    "M. equilateral λ/2 triangle tilted 45° about E axis": (
        np.array([[0, np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))],
                  [np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60)) * np.cos(np.deg2rad(45)),
                   np.cos(np.deg2rad(60)) * np.sin(np.deg2rad(45))]]),
        [0.5, 0.5],
    ),
    "O. tilted L: N λ/2 horizontal + (E+Up)/√2 λ/2 (3 antennas)": (
        np.array([[0, 1, 0.0], [np.sqrt(0.5), 0, np.sqrt(0.5)]]),
        [0.5, 0.5],
    ),
    "P. tilted L: N λ horizontal + (E+Up)/√2 λ/2 (3 antennas)": (
        np.array([[0, 1, 0.0], [np.sqrt(0.5), 0, np.sqrt(0.5)]]),
        [1.0, 0.5],
    ),
    "N. L-shape λ/2 horizontal + Up λ/2 (4 antennas, 3-D)": (
        np.array([[0, 1, 0.0], [1, 0, 0.0], [0, 0, 1.0]]),
        [0.5, 0.5, 0.5],
    ),
}


def ambiguity_table():
    print("\n=== 1. Ambiguity: number of physical (az, el) candidates, sky grid az 0..359 step 5, el 2..88 step 2 ===")
    print(f"{'geometry':62s} {'unique':>7s} {'2':>5s} {'3':>5s} {'>=4':>5s}  | unique el<=15° | unique el 30-60° | el<=15° unique with prior el<=25°")
    azs = np.arange(0, 360, 5.0)
    els = np.arange(2, 89, 2.0)
    for name, (B, k) in GEOMETRIES.items():
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        low_unique = low_n = mid_unique = mid_n = low_prior_unique = 0
        for az in azs:
            for el in els:
                u = unit(az, el)
                cands = array_candidates(B, np.array(k), u)
                n = len(cands)
                counts[min(n, 4)] += 1
                if el <= 15:
                    low_n += 1
                    low_unique += n == 1
                    low_prior_unique += sum(1 for v in cands if az_el(v)[1] <= 25.0) == 1
                if 30 <= el <= 60:
                    mid_n += 1
                    mid_unique += n == 1
        tot = sum(counts.values())
        print(
            f"{name:62s} {100*counts[1]/tot:6.1f}% {100*counts[2]/tot:4.1f}% {100*counts[3]/tot:4.1f}% {100*counts[4]/tot:4.1f}%"
            f"  | {100*low_unique/low_n:6.1f}%        | {100*mid_unique/mid_n:6.1f}%          | {100*low_prior_unique/low_n:6.1f}%"
        )
    print("(geometry A counts cone candidates, i.e. the two Eq. 21 branches; az/el are not defined for one baseline)")


def example_candidates():
    print("\n=== 2. Worked example: true source az=40°, el=10° (ground spoofer), L-shape both D=λ ===")
    B, k = GEOMETRIES["B. L-shape, both D=λ"]
    u = unit(40, 10)
    for v in array_candidates(B, np.array(k), u):
        az, el = az_el(v)
        print(f"   candidate az={az:6.1f}°  el={el:5.1f}°   c_N={v[1]:+.3f} c_E={v[0]:+.3f}")
    print("   same source, L-shape D=λ (N) and D=λ/2 (E):")
    B, k = GEOMETRIES["D. L-shape, D=λ and D=λ/2"]
    for v in array_candidates(B, np.array(k), u):
        az, el = az_el(v)
        print(f"   candidate az={az:6.1f}°  el={el:5.1f}°")
    print("   same source, both D=λ/2:")
    B, k = GEOMETRIES["E. L-shape, both D=λ/2"]
    for v in array_candidates(B, np.array(k), u):
        az, el = az_el(v)
        print(f"   candidate az={az:6.1f}°  el={el:5.1f}°")
    print("   true source az=40°, el=45° (elevated spoofer), L-shape both D=λ:")
    B, k = GEOMETRIES["B. L-shape, both D=λ"]
    for v in array_candidates(B, np.array(k), unit(40, 45)):
        az, el = az_el(v)
        print(f"   candidate az={az:6.1f}°  el={el:5.1f}°")


def noise_table(sigma_f=0.012, n_prn=4, n_epoch=1, trials=4000):
    """Monte Carlo of az/el error for the CORRECT candidate, noise sigma_f cycles per PRN per
    epoch on each baseline's fractional DD, averaged over n_prn PRNs and n_epoch epochs
    (independent noise), reference satellite direction known exactly."""
    print(f"\n=== 3. Noise only (correct candidate), sigma_f={sigma_f} cyc/PRN/epoch, group of {n_prn} PRNs, {n_epoch} epoch(s) ===")
    print(f"{'geometry':62s} " + " ".join(f"el={el:>2d}°: az/el 1σ" for el in (5, 15, 30, 45, 60)))
    sig = sigma_f / np.sqrt(n_prn * n_epoch)
    for name, (B, k) in GEOMETRIES.items():
        if B.shape[0] < 2:
            continue
        k = np.array(k)
        cells = []
        for el in (5, 15, 30, 45, 60):
            az_err, el_err = [], []
            for az in (20, 65, 110, 155, 200, 245, 290, 335):
                u = unit(az, el)
                c_true = B @ u
                for _ in range(trials // 8):
                    c = c_true + rng.normal(0, sig, size=len(k)) / k
                    sols = solve_direction(B, c, require_upper=False, tol=np.inf)
                    if not sols:  # |u_par|>1 from noise: clip to the horizon
                        u_par = np.linalg.pinv(B) @ c
                        sols = [u_par / np.linalg.norm(u_par)]
                    # take the solution nearest the truth (mirror already excluded by geometry)
                    v = min(sols, key=lambda s: np.linalg.norm(s - u))
                    a, e = az_el(v)
                    az_err.append((a - az + 180) % 360 - 180)
                    el_err.append(e - el)
            cells.append(f"{np.std(az_err):5.2f}/{np.std(el_err):5.2f}°")
        print(f"{name:62s} " + "   ".join(cells))


def bias_table(bias_cyc=0.03):
    """A common bias of bias_cyc on each baseline (antenna phase-centre offset or baseline
    calibration error that does NOT average over PRNs or epochs) -> az/el shift."""
    print(f"\n=== 4. Systematic bias of {bias_cyc} cyc on each baseline (does not average out): az/el shift ===")
    print(f"{'geometry':62s} " + " ".join(f"el={el:>2d}°: Δaz/Δel" for el in (5, 15, 30, 45, 60)))
    for name, (B, k) in GEOMETRIES.items():
        if B.shape[0] < 2:
            continue
        k = np.array(k)
        cells = []
        for el in (5, 15, 30, 45, 60):
            worst_az = worst_el = 0.0
            for az in range(0, 360, 15):
                u = unit(az, el)
                for signs in itertools.product((-1, 1), repeat=len(k)):
                    c = B @ u + np.array(signs) * bias_cyc / k
                    sols = solve_direction(B, c, require_upper=False, tol=np.inf)
                    if not sols:
                        u_par = np.linalg.pinv(B) @ c
                        sols = [u_par / np.linalg.norm(u_par)]
                    v = min(sols, key=lambda s: np.linalg.norm(s - u))
                    a, e = az_el(v)
                    worst_az = max(worst_az, abs((a - az + 180) % 360 - 180))
                    worst_el = max(worst_el, abs(e - el))
            cells.append(f"{worst_az:5.1f}/{worst_el:5.1f}°")
        print(f"{name:62s} " + "   ".join(cells))


if __name__ == "__main__":
    ambiguity_table()
    example_candidates()
    noise_table(sigma_f=0.012, n_prn=1, n_epoch=1)
    noise_table(sigma_f=0.012, n_prn=4, n_epoch=100)
    bias_table(0.03)
