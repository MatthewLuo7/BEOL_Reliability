from scipy.integrate import quad
from scipy.integrate import IntegrationWarning
from scipy.special import erf
import numpy as np
from scipy.special import erfcinv
from scipy.optimize import root_scalar
import warnings
import VERIFIED_Unit_Level_DPM_Based as tbd

from Global_Params import N_VIA, T_TARGET, TOTAL_LINE_LENGTH_NM, RHO_LER, F_TARGET

# Total number of vias assumed in the paper for 3km interconnect


# ==========================================
# Via-to-Line Statistical Distributions
# ==========================================

def pdf_integrand_via_dominated(ler_R, elsr, sigma_ler, sigma_via, rho_ler=RHO_LER):
    """
    Equation (A2): Integrand for the probability distribution of ELSR 
    when the Via Misalignment dominates the left line-edge roughness (vm > ler_L).
    """
    # Safeguard against zero-division
    sigma_ler = max(sigma_ler, 1e-9)
    sigma_via = max(sigma_via, 1e-9)
    
    term1 = np.exp(-(ler_R**2) / (2 * sigma_ler**2))

    term2 = np.exp(-((elsr + ler_R)**2) / (2 * sigma_via**2))
    
    erf_input_num = (elsr + ler_R * (1 - rho_ler))

    # The paper has a typo where they write sigma_via instead of sigma_ler in the denominator of the erf argument.
    erf_input_den = sigma_ler * np.sqrt(2 * (1 - rho_ler**2))

    erf_input = erf_input_num / erf_input_den
    term3 = 1 + erf(erf_input)
    
    denominator = 4 * np.pi * sigma_ler * sigma_via
    return (term1 * term2 * term3) / denominator

def pdf_via_dominated(elsr, sigma_ler, sigma_via, rho_ler=RHO_LER):
    """
    Equation (A2): PDF of ELSR when via misalignment dominates.
    Integrates the via-dominated integrand across all possible ler_R values.
    """
    # Integrate over a reasonable range of ler_R (-5 sigma to +5 sigma)
    limit = 5 * max(sigma_ler, sigma_via)


    res, _ = quad(pdf_integrand_via_dominated, -limit, limit, args=(elsr, sigma_ler, sigma_via, rho_ler), limit=200)
    return res

def pdf_integrand_line_dominated(ler_R, elsr, sigma_ler, sigma_via, rho_ler=RHO_LER):
    """
    Equation (A4): Integrand for the probability distribution of ELSR 
    when the left line-edge roughness dominates Via Misalignment (ler_L > vm).
    """
    sigma_ler = max(sigma_ler, 1e-9)
    sigma_via = max(sigma_via, 1e-9)
    

    exponent_num =((elsr+ler_R)**2 + ler_R**2 - 2 * (elsr + ler_R) * ler_R * rho_ler)
    exponent_demoninator = 2*(rho_ler**2 - 1) * sigma_ler**2
    
    exponent = exponent_num / exponent_demoninator


    term1 = np.exp(exponent)
    
    erf_arg = (elsr+ler_R) / (np.sqrt(2) * sigma_via)

    term2 = 1 + erf(erf_arg)
    
    denominator = 4 * np.pi * (sigma_ler**2) * np.sqrt(1 - rho_ler**2)
    return (term1 * term2) / denominator

def pdf_line_dominated(elsr, sigma_ler, sigma_via, rho_ler=RHO_LER):
    """
    Equation (A4): PDF of ELSR when line edge roughness dominates.
    Integrates the line-dominated integrand across all possible ler_R values.
    """
    # Integrate over a reasonable range of ler_R (-5 sigma to +5 sigma)
    limit = 5 * max(sigma_ler, sigma_via)
    res, _ = quad(pdf_integrand_line_dominated, -limit, limit, args=(elsr, sigma_ler, sigma_via, rho_ler), limit=200)
    return res

# ==========================================
# Handling F_(V>L) and F_(V<L) Together with Safe Numerical Integration
# ==========================================

def calc_pdf_elsr(elsr, sigma_ler, sigma_via, rho_ler=RHO_LER):
    """
    Calculates the combined Probability Density Function (PDF) value 
    for a given Effective Local Spacing Roughness (ELSR).
    """

    # Calculate the probability of via-dominated vs line-dominated scenarios

    pdf_elsr = pdf_via_dominated(elsr, sigma_ler, sigma_via, rho_ler) + pdf_line_dominated(elsr, sigma_ler, sigma_via, rho_ler)

    return pdf_elsr

def failure_term(S_min, t_target):
    """
    Compute (t/eta)^beta for a given minimum spacing.
    Returns (failure_term, is_valid).
    """
    if S_min <= 1.0:
        return 0.0, True  # yield issue, not reliability

    beta = tbd.calc_beta_tBD(S_min)
    eta = tbd.calc_eta_tBD(S_min)

    if eta <= 0 or not np.isfinite(eta) or beta <= 0 or not np.isfinite(beta):
        return np.inf, False

    log_ft = beta * (np.log(t_target) - np.log(eta))

    if log_ft > 700:
        return np.inf, False
    if log_ft < -745:
        return 0.0, True
    return np.exp(log_ft), True

def integrand_via_dominated(elsr, s_nom, sigma_ler, sigma_via, rho_ler=RHO_LER, t_target=T_TARGET):
    """Failure contribution weighted by the via-dominated ELSR PDF."""
    ft, ok = failure_term(s_nom - elsr, t_target)
    if not ok:
        return np.inf
    pdf_val = pdf_via_dominated(elsr, sigma_ler, sigma_via, rho_ler)
    return N_VIA * pdf_val * ft


def integrand_line_dominated(elsr, s_nom, sigma_ler, sigma_via, rho_ler=RHO_LER, t_target=T_TARGET):
    """Failure contribution weighted by the line-dominated ELSR PDF."""
    ft, ok = failure_term(s_nom - elsr, t_target)
    if not ok:
        return np.inf
    pdf_val = pdf_line_dominated(elsr, sigma_ler, sigma_via, rho_ler)
    return N_VIA * pdf_val * ft

# ==========================================
# Benard bounds (Equations 12-13)
# ==========================================

def _benard_limit(sigma, N):
    """Worst-case deviation via Benard's median-rank approximation."""
    erf_arg = (2 * (1 - 0.3)) / (N + 0.4)
    return np.sqrt(2) * sigma * erfcinv(erf_arg)

# ==========================================
# Main F_V calculation with separated integrals
# ==========================================

def _integrate_safely(integrand_fn, elsr_min, elsr_max, args):
    """
    Integrate with boundary-divergence check.
    Returns the integral value, or np.inf if divergent.
    """
    if elsr_min >= elsr_max:
        return 0.0

    # Check whether the integrand is manageable near the upper bound
    probe = integrand_fn(elsr_max - 0.01, *args)
    if not np.isfinite(probe) or probe > 1e30:
        return np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            val, _ = quad(
                integrand_fn, elsr_min, elsr_max,
                args=args, limit=300, epsabs=1e-20, epsrel=1e-10,
            )
        except Exception:
            return np.inf

    return val if (np.isfinite(val) and val >= 0) else np.inf

def calc_F_V_greater_than_L(s_nom, sigma_ler, sigma_via, rho_ler=RHO_LER, t_target=T_TARGET):
    """
    Calculates only the failure probability for the subset where via 
    misalignment dominates line edge roughness (F_{V>L}).
    """
    yield_limit = s_nom - 1.001
    args_common = (s_nom, sigma_ler, sigma_via, rho_ler, t_target)

    # ---- Via-dominated integral ONLY ----
    sigma_eff_via = np.sqrt(sigma_via ** 2 + sigma_ler ** 2)

    elsr_max_via = min(_benard_limit(sigma_eff_via, N_VIA), yield_limit)
    elsr_min_via = -_benard_limit(sigma_eff_via, N_VIA)
    
    I_via = _integrate_safely(integrand_via_dominated, elsr_min_via, elsr_max_via, args_common)

    if not np.isfinite(I_via):
        return 1.0

    return 1 - np.exp(-I_via)

def calc_F_V_less_than_L(s_nom, sigma_ler, sigma_via, rho_ler=RHO_LER, t_target=T_TARGET):
    """
    Calculates only the failure probability for the subset where line edge 
    roughness dominates via misalignment (F_{V<L}).
    """
    yield_limit = s_nom - 1.001
    args_common = (s_nom, sigma_ler, sigma_via, rho_ler, t_target)

    # ---- Line-dominated integral ONLY ----
    sigma_lsr = np.sqrt(2 * (1 - rho_ler)) * sigma_ler
    elsr_max_line = min(_benard_limit(sigma_lsr, TOTAL_LINE_LENGTH_NM), yield_limit)
    elsr_min_line = -_benard_limit(sigma_lsr, TOTAL_LINE_LENGTH_NM)
  
    I_line = _integrate_safely(integrand_line_dominated, elsr_min_line, elsr_max_line, args_common)

    if not np.isfinite(I_line):
        return 1.0

    return 1 - np.exp(-I_line)

def calc_F_V(s_nom, sigma_ler, sigma_via, rho_ler=RHO_LER, t_target=T_TARGET):
    """
    Equation (8): Total via-to-line failure probability.

    Via-dominated and line-dominated contributions are integrated SEPARATELY
    with their own Benard bounds:

      Via-dominated (vm > ler_L):
        ELSR = vm - ler_R  (vm and ler_R independent)
        Effective σ = sqrt(σ_via² + σ_LER²)
        Sample count = N_VIA

      Line-dominated (ler_L > vm):
        ELSR = ler_L - ler_R  (correlated with coefficient ρ)
        Effective σ = σ_LSR = sqrt(2(1-ρ)) · σ_LER
        Sample count = L_line  (in nm, same unit as σ_LER)

    F_V = 1 - exp(-(I_via_dom + I_line_dom))
    """
    yield_limit = s_nom - 1.001
    args_common = (s_nom, sigma_ler, sigma_via, rho_ler, t_target)

    # ---- Via-dominated integral ----
    # sigma_eff_via = np.sqrt(sigma_via ** 2 + sigma_ler ** 2)

    # elsr_max_via = min(_benard_limit(sigma_eff_via, N_VIA), yield_limit)
    # elsr_min_via = -_benard_limit(sigma_eff_via, N_VIA)

    sigma_eff_via = np.sqrt(sigma_via ** 2 + sigma_ler ** 2)

    elsr_max_via = min(_benard_limit(sigma_eff_via, N_VIA), yield_limit)
    elsr_min_via = -_benard_limit(sigma_eff_via, N_VIA)

    # print(f"Via-dominated ELSR bounds: [{elsr_min_via:.3f}, {elsr_max_via:.3f}]")
    # print(f"Via-dominated integrand at upper bound: {integrand_via_dominated(elsr_max_via - 0.01, *args_common):.3e}")
    # print(f"Via-dominated integrand at lower bound: {integrand_via_dominated(elsr_min_via + 0.01, *args_common):.3e}")
    
    I_via = _integrate_safely(integrand_via_dominated, elsr_min_via, elsr_max_via, args_common)

    # ---- Line-dominated integral ----
    sigma_lsr = np.sqrt(2 * (1 - rho_ler)) * sigma_ler
    elsr_max_line = min(_benard_limit(sigma_lsr, TOTAL_LINE_LENGTH_NM), yield_limit)
    elsr_min_line = -_benard_limit(sigma_lsr, TOTAL_LINE_LENGTH_NM)
  
    I_line = _integrate_safely(integrand_line_dominated, elsr_min_line, elsr_max_line, args_common)

    # ---- Combine ----
    I_total = I_via + I_line

    if not np.isfinite(I_total):
        return 1.0

    return 1 - np.exp(-I_total)


def find_sigma_via_for_target_F_V(s_nom, sigma_ler, rho_ler=RHO_LER, t_target=T_TARGET, f_target=F_TARGET):
    """
    Uses root_scalar to find the sigma_via that achieves the target F_V.
    Assumes F_V is monotonic in sigma_via and auto-finds a valid bracket.
    """
    def objective(sigma_via):
        return calc_F_V(s_nom, sigma_ler, sigma_via, rho_ler, t_target) - f_target

    # Lower anchor near zero variation.
    a = 1e-9
    fa = objective(a)
    if fa >= 0:
        return a

    # Expand upper bound until sign flips (fa < 0, fb > 0).
    b = 1e-3
    fb = objective(b)
    max_b = 1e6
    max_iterations = 80
    for _ in range(max_iterations):
        if fb >= 0:
            break
        b *= 2.0
        if b > max_b:
            break
        fb = objective(b)

    if fb < 0:
        raise ValueError(
            "Cannot bracket solution: target F_V is larger than the computed F_V "
            f"even at sigma_via={b:.2e}. objective(a={a:.2e})={fa:.2e}, objective(b={b:.2e})={fb:.2e}"
        )

    result = root_scalar(objective, bracket=[a, b], method='bisect')
    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge")

if __name__ == "__main__":
    S_nom = 6
    t = T_TARGET
    f_v = calc_F_V(S_nom, sigma_ler=0.0001, sigma_via=0.8, rho_ler=0.8, t_target=t)
    print(f"F_V(SNOM={S_nom}, sigma_ler=0.0001, sigma_via=0.8) = {f_v:.6e}")
    S_nom = 6
    t = T_TARGET
    f_v = calc_F_V(S_nom, sigma_ler=0.0001, sigma_via=2, rho_ler=0.8, t_target=t)
    print(f"F_V(SNOM={S_nom}, sigma_ler=0.0001, sigma_via=2) = {f_v:.6e}")

    # Example of finding sigma_via for a target F_V
    target_f_v = 100 / 1e6  # 100 ppm
    try:
        sigma_via_needed = find_sigma_via_for_target_F_V(S_nom, sigma_ler=0.01, rho_ler=0.8, t_target=t, f_target=target_f_v)
        print(f"Sigma_via needed for F_V={target_f_v:.2e}: {sigma_via_needed:.3f}")
    except ValueError as e:
        print(str(e))

    try:
        sigma_via_needed = find_sigma_via_for_target_F_V(8, sigma_ler=0.01, rho_ler=0.8, t_target=t, f_target=target_f_v)
        print(f"Sigma_via needed for F_V={target_f_v:.2e}: {sigma_via_needed:.3f}")
    except ValueError as e:
        print(str(e))

