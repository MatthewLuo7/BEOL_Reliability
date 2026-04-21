"""
Generates Figure 7 from Paper 2:
"Three-Dimensional Modeling of BEOL TDDB Variability Specs for Sub-20nm Half-Pitch Interconnects"

Figure 7(a): Maximum allowed 1σ variation vs S_NOM
Figure 7(b): Calculated variation limit contours (σ_LER vs σ_via)

Equation (7): F_WS = 1 - (1 - F_T)(1 - F_L)(1 - F_V)

The contour in 7(b) solves F_total = 1 - (1-F_L)(1-F_V) = F_TARGET
for each (S_NOM, σ_LER) pair, finding the maximum allowed σ_via.
Tips (F_T) are excluded per the paper's analysis that tip-to-tip
TDDB is negligible when S_TNOM = 2 * S_NOM.
"""
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt

import numpy as np
import json

from Failure_Via_Bounds_Seperated import calc_F_V, calc_F_V_greater_than_L, calc_F_V_less_than_L
from VERIFIED_Failure_Line import calc_F_L
from Global_Params import F_TARGET, SIGMA_DIE, RHO_LER


# ==========================================
# Figure 7(a): Individual variation specs
# ==========================================

def find_max_sigma_via_only(s_nom, rho_ler=RHO_LER):
    """
    Fig 7(a) gray curve.
    Structures: straight lines without LER, one via per side.
    Solve: F_V(s_nom, σ_LER≈0, σ_via) = F_TARGET
    """
    lo, hi = 0.001, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        fv = calc_F_V(s_nom, 0.01, mid, rho_ler)
        if fv > F_TARGET:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.001:
            break
    return (lo + hi) / 2

def find_max_sigma_via_gt_ler(s_nom, rho_ler=RHO_LER):
    """
    Fig 7(a) gray curve.
    Structures: straight lines with LER, one via per side.
    Solve: F_V(s_nom, σ_LER=σ_via, σ_via) = F_TARGET
    """
    lo, hi = 0.001, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        fv = calc_F_V_greater_than_L(s_nom, 0.01, mid, rho_ler)
        if fv > F_TARGET:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.001:
            break
    return (lo + hi) / 2

def find_max_sigma_via_lt_ler(s_nom, rho_ler=RHO_LER):
    """
    Fig 7(a) gray curve.
    Structures: straight lines with LER, one via per side.
    Solve: F_V(s_nom, σ_LER=σ_via/2, σ_via) = F_TARGET
    """
    lo, hi = 0.001, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        fv = calc_F_V_less_than_L(s_nom, mid, 0.01, rho_ler)
        if fv > F_TARGET:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.001:
            break
    return (lo + hi) / 2


def find_max_sigma_ler_only(s_nom, rho_ler=RHO_LER, sigma_die=SIGMA_DIE):
    """
    Fig 7(a) black curves.
    Structures: lines without vias.
    σ_WLSR = sqrt(2(1-ρ)σ_LER² + σ_die²)
    Solve: F_L(σ_WLSR, s_nom) = F_TARGET
    """
    lo, hi = 0.001, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        sigma_wlsr = np.sqrt(2 * (1 - rho_ler) * mid ** 2 + sigma_die ** 2)
        fl = calc_F_L(s_nom, sigma_wlsr)[0]
        if fl > F_TARGET:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.001:
            break
    return (lo + hi) / 2


# ==========================================
# Figure 7(b): Combined contour
# ==========================================

def calc_F_total(s_nom, sigma_ler, sigma_via, rho_ler=RHO_LER, sigma_die=SIGMA_DIE):
    """
    Equation (7) without tips:
    F_total = 1 - (1 - F_L)(1 - F_V)
    """
    sigma_wlsr = np.sqrt(2 * (1 - rho_ler) * sigma_ler ** 2 + sigma_die ** 2)
    fl = calc_F_L(s_nom, sigma_wlsr)[0]
    fv = calc_F_V(s_nom, sigma_ler, sigma_via, rho_ler)
    return 1 - (1 - fl) * (1 - fv)


def find_contour_point(s_nom, sigma_ler, rho_ler=RHO_LER, sigma_die=SIGMA_DIE):
    """
    For given S_NOM and σ_LER, find max σ_via such that
    F_total = 1 - (1-F_L)(1-F_V) = F_TARGET.
    """
    # Check whether F_L alone already exceeds target
    sigma_wlsr = np.sqrt(2 * (1 - rho_ler) * sigma_ler ** 2 + sigma_die ** 2)
    fl = calc_F_L(s_nom, sigma_wlsr)[0]
    if fl >= F_TARGET:
        return np.nan

    lo, hi = 0.01, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        ft = calc_F_total(s_nom, sigma_ler, mid, rho_ler, sigma_die)
        if ft > F_TARGET:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.003:
            break

    result = (lo + hi) / 2
    return result if result > 0.02 else np.nan


def compute_figure7a_data(snom_range):
    """Compute Figure 7(a) data dictionary."""
    print("Computing Figure 7(a)...")
    fig7a = {"snom": [], "vm": [], "ler_mp": [], "ler_sp": [], "vm_gt_ler": [], "vm_lt_ler": []}

    for s_nom in snom_range:
        fig7a["snom"].append(float(s_nom))
        fig7a["vm"].append(float(find_max_sigma_via_only(s_nom)))
        fig7a["ler_mp"].append(float(find_max_sigma_ler_only(s_nom, rho_ler=0.8)))
        fig7a["ler_sp"].append(float(find_max_sigma_ler_only(s_nom, rho_ler=0.0)))
        fig7a["vm_gt_ler"].append(float(find_max_sigma_via_gt_ler(s_nom, rho_ler=0.8)))
        fig7a["vm_lt_ler"].append(float(find_max_sigma_via_lt_ler(s_nom, rho_ler=0.8)))
        print(
            f"  S_NOM={s_nom:.1f}: VM={fig7a['vm'][-1]:.3f}, "
            f"LER_MP={fig7a['ler_mp'][-1]:.3f}, LER_SP={fig7a['ler_sp'][-1]:.3f},VM_GT_LER={fig7a['vm_gt_ler'][-1]:.3f}, VM_LT_LER={fig7a['vm_lt_ler'][-1]:.3f}"
        )

    return fig7a


def compute_figure7b_data(S_targets, ler_range):
    """Compute Figure 7(b) contour data dictionary."""
    print("\nComputing Figure 7(b) contours...")
    fig7b = {}

    for S in S_targets:
        contour_ler, contour_via = [], []
        for ler in ler_range:
            via = find_contour_point(S, ler)
            if np.isfinite(via):
                contour_ler.append(float(ler))
                contour_via.append(float(via))
            else:
                break
        fig7b[str(S)] = {"ler": contour_ler, "via": contour_via}
        print(f"  S_NOM={S}: {len(contour_ler)} contour points")

    return fig7b


def plot_figure7a(fig7a):
    """Plot Figure 7(a): maximum allowed 1σ variation vs S_NOM."""
    snom = np.array(fig7a["snom"])
    vm = np.array(fig7a["vm"])
    ler_mp = np.array(fig7a["ler_mp"])
    ler_sp = np.array(fig7a["ler_sp"])

    plt.figure(figsize=(7, 5))
    plt.plot(snom, vm, color="gray", linewidth=2.5, label=r"$\sigma_{via}$ (via-only)")
    plt.plot(snom, ler_mp, color="black", linewidth=2.5, linestyle="-", label=r"$\sigma_{LER}$, MP ($\rho=0.8$)")
    plt.plot(snom, ler_sp, color="black", linewidth=2.5, linestyle="--", label=r"$\sigma_{LER}$, SP ($\rho=0.0$)")
    plt.plot(snom, fig7a["vm_gt_ler"], color="blue", linewidth=2.5, linestyle="-.", label=r"$\sigma_{via}$ (via > LER)")
    plt.plot(snom, fig7a["vm_lt_ler"], color="red", linewidth=2.5, linestyle="-.", label=r"$\sigma_{via}$ (via < LER)")

    plt.xlim(0, float(np.max(snom)))
    ymax = max(np.max(vm), np.max(ler_mp), np.max(ler_sp))
    plt.ylim(0, max(2.0, 1.1 * float(ymax)))
    plt.xlabel(r"$S_{NOM}$ [nm]", fontsize=12)
    plt.ylabel(r"Maximum allowed 1$\sigma$ variation [nm]", fontsize=12)
    plt.title("Figure 7(a): Maximum allowed variation", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_figure7b(fig7b, S_targets):
    """Plot Figure 7(b): calculated variation limit contours."""
    plt.figure(figsize=(7, 5))
    cmap = plt.get_cmap('viridis', max(S_targets) + 1)

    for S_str, values in fig7b.items():
        S = int(S_str)
        ler_vals = values["ler"]
        via_vals = values["via"]

        if len(ler_vals) > 0:
            line, = plt.plot(ler_vals, via_vals, color=cmap(S), linewidth=2)

            lbl_idx = min(1, len(ler_vals) - 1)
            plt.text(
                ler_vals[lbl_idx], via_vals[lbl_idx] + 0.05, str(S),
                color=line.get_color(), fontsize=10, fontweight='bold'
            )

    ratio_x = np.linspace(0, 1.0, 10)
    plt.plot(
        ratio_x, ratio_x * 2, color='gray', linestyle='--',
        linewidth=1, marker='o', markerfacecolor='none'
    )

    plt.xlim(0, 2.0)
    plt.ylim(0, 2.0)
    plt.xlabel(r'$\sigma_{LER}$ [nm]', fontsize=12)
    plt.ylabel(r'$\sigma_{via}$ [nm]', fontsize=12)
    plt.title('Figure 7(b): Calculated variation limit contours', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=4, vmax=14))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), ticks=[4, 6, 8, 10, 12, 14])
    cbar.set_label(r'S$_{NOM}$ [nm]')

    plt.tight_layout()
    plt.show()


def parse_requested_figures(argv):
    """Parse optional CLI arg: a, b, or both (default)."""
    if len(argv) < 2:
        return "both"

    req = argv[1].strip().lower()
    if req in {"a", "b", "both"}:
        return req

    raise ValueError("Usage: python Plot_Figure7.py [a|b|both]")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    requested = parse_requested_figures(sys.argv)

    snom_range = np.arange(4, 21, 0.5)
    S_targets = [4, 5, 6, 7, 8, 9, 10]
    ler_range = np.arange(0.02, 2.0, 0.04)


    data = {}

    fig_num = 1

    if fig_num == 1 or requested in {"a", "both"}:
        fig7a = compute_figure7a_data(snom_range)
        data["fig7a"] = fig7a
        plot_figure7a(fig7a)

    if fig_num == 2 or requested in {"b", "both"}:
        fig7b = compute_figure7b_data(S_targets, ler_range)
        data["fig7b"] = fig7b
        plot_figure7b(fig7b, S_targets)


    with open("fig7_data_v2.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved selected figure data to fig7_data_v2.json (mode={requested})")
