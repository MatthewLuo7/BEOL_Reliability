import numpy as np
import warnings
from VERIFIED_Unit_Level_DPM_Based import calc_eta_tBD, calc_beta_tBD
from scipy.stats import norm
from scipy.integrate import quad
from scipy.integrate import IntegrationWarning
from scipy.special import erfcinv

from Global_Params import T_TARGET, TOTAL_LINE_LENGTH_NM, TOTAL_AREA_NM2, F_TARGET

# ==========================================
# Numerical Integration & Optimization
# ==========================================

def integrand(S, s_nom, sigma_wlsr, t_target=T_TARGET):
    """Calculates the failure probability contribution of a local unit spacing S."""

    # Very small spacing means near-certain local failure and unstable model terms.
    if S <= 0.1:
        return np.inf
    if t_target <= 0:
        return 0.0
    if not np.isfinite(t_target):
        return np.inf

    # PDF for a having a unit with a given spacing S, mean s_nom, and variation sigma_wlsr
    w_S = norm.pdf(S, loc=s_nom, scale=sigma_wlsr)
    beta = calc_beta_tBD(S)
    eta = calc_eta_tBD(S)

    if eta <= 0 or not np.isfinite(eta) or beta <= 0 or not np.isfinite(beta):
        return np.inf

    # Use log(t) - log(eta) instead of log(t/eta) to avoid overflow in the ratio.
    log_failure_term = beta * (np.log(t_target) - np.log(eta))

    if not np.isfinite(log_failure_term):
        return np.inf
    if log_failure_term > 700:
        return np.inf
    if log_failure_term < -745:
        failure_term = 0.0
    else:
        failure_term = np.exp(log_failure_term)
    
    return TOTAL_AREA_NM2 * w_S * failure_term

def target_function(sigma_wlsr, s_nom, f_target=F_TARGET):
    """Difference between the computed integral and the target limit."""
    prob_term = 2 * (1 - 0.3) / (TOTAL_LINE_LENGTH_NM + 0.4)
    s_fs = s_nom - np.sqrt(2) * sigma_wlsr * erfcinv(prob_term)
    
    # GUARD: If variation causes the trench to physically touch (S <= 0.1nm), 
    # the wire is shorted. Return a massive positive value to tell the solver 
    # that the failure rate is way too high and it needs a smaller sigma_wlsr.
    if s_fs <= 0.1:
        return 1.0  
    # Use 5 sigma rule
    upper_bound = s_nom + 5 * sigma_wlsr

    # integrate integrand 
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=IntegrationWarning)
            integral_val, _ = quad(integrand, s_fs, upper_bound, args=(s_nom, sigma_wlsr), limit=200)
    except (IntegrationWarning, RuntimeError, FloatingPointError, OverflowError, ValueError):
        return 1.0

    if not np.isfinite(integral_val) or integral_val < 0:
        return 1.0

    #-I = ln(1 - F_{total})
    #I = -ln(1 - F_{total})
    target_integral = -np.log(1 - f_target)
    
    return integral_val - target_integral

def calc_F_L(s_nom, sigma_wlsr, t_target=T_TARGET):
    """Calculates the overall reliability (failure probability) for a given nominal spacing and sigma_WLSR.
    """

    # Equation (16) S_fs is lower bound of integral.
    prob_term = 2 * (1 - 0.3) / (TOTAL_LINE_LENGTH_NM + 0.4)
    s_fs = s_nom - np.sqrt(2) * sigma_wlsr * erfcinv(prob_term)
    
    if s_fs <= 0.1:
        return 1.0, np.inf  # Shorted, reliability is 0 (failure probability is 100%).
    
    upper_bound = s_nom + 5 * sigma_wlsr

    # Retry with safer numerical settings to reduce isolated integration failures.
    quad_attempts = [
        {"limit": 200, "epsabs": 1.49e-8, "epsrel": 1.49e-8},
        {"limit": 600, "epsabs": 1.0e-10, "epsrel": 1.0e-8},
        {"limit": 1200, "epsabs": 1.0e-12, "epsrel": 1.0e-9},
    ]

    integral_val = np.inf
    for attempt in quad_attempts:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=IntegrationWarning)
                integral_val, _ = quad(
                    integrand,
                    s_fs,
                    upper_bound,
                    args=(s_nom, sigma_wlsr, t_target),
                    limit=attempt["limit"],
                    epsabs=attempt["epsabs"],
                    epsrel=attempt["epsrel"],
                )
            if np.isfinite(integral_val) and integral_val >= 0:
                break
        except (IntegrationWarning, RuntimeError, FloatingPointError, OverflowError, ValueError):
            continue
    else:
        return 1.0, np.inf

    if not np.isfinite(integral_val) or integral_val < 0:
        return 1.0, np.inf
    
    # Convert integral to failure probability
    F_total = 1 -  np.exp(-integral_val)
    return F_total, integral_val
    
    
    
