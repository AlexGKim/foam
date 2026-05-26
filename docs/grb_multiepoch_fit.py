"""
Multi-epoch lever-arm fit for GRB 221009A foam constraint.

Uses Zhang et al. (2024, arXiv:2403.12851v2) Table A3 data:
seven time intervals with significant emission-line detections.
Fits the model r_obs^2(E) = r_int^2 + r_foam,0^2 * f(E/E_0)
for both quasi-static (f = (E/E0)^2) and dynamical (f = (E0/E)^2) regimes.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize

# --- Data from Zhang et al. Table A3 ---
# (time_range, E_line [keV], dE [keV], sigma [keV], dsigma [keV])
# Selected intervals with significant line detections
data_table = [
    ("246-256", 37159, 5340, 7087, 3101),
    ("275-280", 17839, 760, 1667, 720),
    ("285-290", 15016, 536, 1934, 689),
    ("290-295", 12067, 311, 1444, 356),
    ("295-300", 12445, 288, 830, 387),
    ("300-310", 8871, 489, 1740, 529),
    ("310-320", 9726, 256, 1115, 290),
]

labels = [d[0] for d in data_table]
E_keV = np.array([d[1] for d in data_table], dtype=float)
dE_keV = np.array([d[2] for d in data_table], dtype=float)
sig_keV = np.array([d[3] for d in data_table], dtype=float)
dsig_keV = np.array([d[4] for d in data_table], dtype=float)

# Compute r = sigma/E and propagated error
r_obs = sig_keV / E_keV
dr_obs = r_obs * np.sqrt((dsig_keV / sig_keV)**2 + (dE_keV / E_keV)**2)

print("=" * 70)
print("DATA: Zhang et al. (2024) Table A3 — GRB 221009A emission line")
print("=" * 70)
print(f"{'Interval':<12} {'E [MeV]':>10} {'sigma [MeV]':>12} {'r=sig/E':>8} {'dr':>8}")
print("-" * 70)
for i in range(len(labels)):
    print(f"{labels[i]:<12} {E_keV[i]/1e3:>10.2f} {sig_keV[i]/1e3:>12.2f} "
          f"{r_obs[i]:>8.4f} {dr_obs[i]:>8.4f}")
print("-" * 70)

# Weighted mean as sanity check
w = 1.0 / dr_obs**2
r_mean = np.sum(w * r_obs) / np.sum(w)
chi2_const = np.sum(w * (r_obs - r_mean)**2)
print(f"\nWeighted mean r = {r_mean:.4f}")
print(f"chi2 (constant model, dof=6): {chi2_const:.2f}")
print(f"chi2/dof = {chi2_const/6:.2f}")

# --- Physical constants ---
D_C = 1.93e25       # comoving distance [m]
ell_P = 1.616e-35   # Planck length [m]
hc_keV_m = 1.2398e-9  # hc in keV*m (hc = 1.2398e-6 eV*m = 1.2398e-9 keV*m)
E0_keV = 37159.0    # reference energy [keV] (highest detection)
lam0 = hc_keV_m / E0_keV  # lambda_obs at 37 MeV

print(f"\n--- Physical parameters ---")
print(f"D_C = {D_C:.3e} m")
print(f"ell_P = {ell_P:.3e} m")
print(f"E_0 = {E0_keV/1e3:.1f} MeV")
print(f"lambda_0 = hc/E_0 = {lam0:.3e} m")
print(f"ln(D_C/lambda_0) = {np.log(D_C/lam0):.2f}")
print(f"ln(D_C/ell_P) = {np.log(D_C/ell_P):.2f}")


# --- Fitting functions ---
def chi2_qs(r_int, r_foam0, E_data, r_data, dr_data, E_ref):
    """Chi-squared for quasi-static model: r^2 = r_int^2 + r_foam0^2*(E/E0)^2"""
    r_model = np.sqrt(r_int**2 + r_foam0**2 * (E_data / E_ref)**2)
    return np.sum(((r_data - r_model) / dr_data)**2)


def chi2_dyn(r_int, r_foam0, E_data, r_data, dr_data, E_ref):
    """Chi-squared for dynamical model: r^2 = r_int^2 + r_foam0^2*(E0/E)^2"""
    r_model = np.sqrt(r_int**2 + r_foam0**2 * (E_ref / E_data)**2)
    return np.sum(((r_data - r_model) / dr_data)**2)


def profile_chi2(r_foam0_val, chi2_func, E_data, r_data, dr_data, E_ref):
    """Minimize chi2 over r_int at fixed r_foam0, return minimum chi2."""
    def neg_func(r_int):
        if r_int < 0:
            return 1e10
        return chi2_func(r_int, r_foam0_val, E_data, r_data, dr_data, E_ref)
    res = minimize_scalar(neg_func, bounds=(0.0, 0.5), method='bounded')
    return res.fun, res.x


def find_upper_limit(chi2_func, E_data, r_data, dr_data, E_ref, delta_chi2=2.71):
    """Find one-sided 95% CL (delta_chi2=2.71, Wilks) or 68% CL (delta_chi2=1) upper limit on r_foam0."""
    # First find the global minimum (at r_foam0=0)
    chi2_min, r_int_best = profile_chi2(0.0, chi2_func, E_data, r_data, dr_data, E_ref)

    # Now scan r_foam0 upward to find where profiled chi2 = chi2_min + delta_chi2
    target = chi2_min + delta_chi2

    # Binary search
    r_lo, r_hi = 0.0, 0.5
    for _ in range(100):
        r_mid = (r_lo + r_hi) / 2
        chi2_val, _ = profile_chi2(r_mid, chi2_func, E_data, r_data, dr_data, E_ref)
        if chi2_val < target:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return (r_lo + r_hi) / 2, chi2_min, r_int_best


def alpha_bound_qs(r_foam0, D_C, lam0, ell_P):
    """Alpha bound for quasi-static: sigma_l = r_foam0 * lam0"""
    sigma_l = r_foam0 * lam0
    # sigma_l^2 = D_C^{2(1-alpha)} * ell_P^{2*alpha}
    # ln(sigma_l) = (1-alpha)*ln(D_C) + alpha*ln(ell_P)
    # alpha = [ln(D_C) - ln(sigma_l)] / [ln(D_C) - ln(ell_P)]
    #       = [ln(D_C/sigma_l)] / ln(D_C/ell_P)
    #       = [ln(D_C/lam0) - ln(r_foam0)] / ln(D_C/ell_P)
    return (np.log(D_C / lam0) - np.log(r_foam0)) / np.log(D_C / ell_P)


def alpha_bound_dyn(r_foam0, D_C, lam0, ell_P):
    """Alpha bound for dynamical: sigma_l = lam0 / (2*pi*r_foam0)"""
    sigma_l = lam0 / (2 * np.pi * r_foam0)
    return (np.log(D_C) - np.log(sigma_l)) / np.log(D_C / ell_P)


# --- Main analysis ---
print("\n" + "=" * 70)
print("FIT RESULTS")
print("=" * 70)

for regime, chi2_func, alpha_func, regime_name in [
    ("Quasi-static (r_foam ∝ E)", chi2_qs, alpha_bound_qs, "qs"),
    ("Dynamical (r_foam ∝ 1/E)", chi2_dyn, alpha_bound_dyn, "dyn"),
]:
    print(f"\n--- {regime} ---")

    # Full dataset
    r_foam_95, chi2_min, r_int_best = find_upper_limit(
        chi2_func, E_keV, r_obs, dr_obs, E0_keV, delta_chi2=2.71)
    r_foam_68, _, _ = find_upper_limit(
        chi2_func, E_keV, r_obs, dr_obs, E0_keV, delta_chi2=1.0)

    print(f"  Best fit: r_int = {r_int_best:.4f}, r_foam,0 = 0 (boundary)")
    print(f"  chi2_min = {chi2_min:.2f} (dof = {len(E_keV)-1})")
    print(f"  68% CL upper limit on r_foam,0: {r_foam_68:.4f}")
    print(f"  95% CL upper limit on r_foam,0: {r_foam_95:.4f}")

    alpha_68 = alpha_func(r_foam_68, D_C, lam0, ell_P)
    alpha_95 = alpha_func(r_foam_95, D_C, lam0, ell_P)
    print(f"  alpha bound (68% CL): alpha >= {alpha_68:.4f}")
    print(f"  alpha bound (95% CL): alpha >= {alpha_95:.4f}")
    print(f"  Compare: single-line bound alpha >= {(np.log(D_C/lam0) - np.log(10))/np.log(D_C/ell_P):.4f}")
    print(f"  Holographic value alpha = 2/3 = {2/3:.4f}")

    # Excluding G1 (246-256s, index 0)
    print(f"\n  --- Excluding G1 (246-256s) ---")
    E_noG1 = E_keV[1:]
    r_noG1 = r_obs[1:]
    dr_noG1 = dr_obs[1:]
    # Reference energy becomes the highest remaining
    E0_noG1 = E_noG1[0]  # 17839 keV
    lam0_noG1 = hc_keV_m / E0_noG1

    r_foam_95_noG1, chi2_min_noG1, r_int_noG1 = find_upper_limit(
        chi2_func, E_noG1, r_noG1, dr_noG1, E0_noG1, delta_chi2=2.71)
    r_foam_68_noG1, _, _ = find_upper_limit(
        chi2_func, E_noG1, r_noG1, dr_noG1, E0_noG1, delta_chi2=1.0)

    alpha_95_noG1 = alpha_func(r_foam_95_noG1, D_C, lam0_noG1, ell_P)
    alpha_68_noG1 = alpha_func(r_foam_68_noG1, D_C, lam0_noG1, ell_P)
    print(f"  Best fit: r_int = {r_int_noG1:.4f}, r_foam,0 = 0")
    print(f"  chi2_min = {chi2_min_noG1:.2f} (dof = {len(E_noG1)-1})")
    print(f"  68% CL upper limit on r_foam,0: {r_foam_68_noG1:.4f}")
    print(f"  95% CL upper limit on r_foam,0: {r_foam_95_noG1:.4f}")
    print(f"  alpha bound (68% CL, no G1): alpha >= {alpha_68_noG1:.4f}")
    print(f"  alpha bound (95% CL, no G1): alpha >= {alpha_95_noG1:.4f}")


# --- Sensitivity: allow linear r_int(E) ---
print("\n" + "=" * 70)
print("SENSITIVITY: Linear r_int(E) = r0 + r1*(E/E0)")
print("=" * 70)

def chi2_qs_linear(params, E_data, r_data, dr_data, E_ref):
    """Quasi-static with linear intrinsic: r^2 = (r0+r1*E/E0)^2 + rf^2*(E/E0)^2"""
    r0, r1, rf = params
    r_int_E = r0 + r1 * (E_data / E_ref)
    r_model = np.sqrt(r_int_E**2 + rf**2 * (E_data / E_ref)**2)
    return np.sum(((r_data - r_model) / dr_data)**2)


def chi2_dyn_linear(params, E_data, r_data, dr_data, E_ref):
    """Dynamical with linear intrinsic: r^2 = (r0+r1*E/E0)^2 + rf^2*(E0/E)^2"""
    r0, r1, rf = params
    r_int_E = r0 + r1 * (E_data / E_ref)
    r_model = np.sqrt(r_int_E**2 + rf**2 * (E_ref / E_data)**2)
    return np.sum(((r_data - r_model) / dr_data)**2)


for regime_name, chi2_lin_func, alpha_func in [
    ("Quasi-static", chi2_qs_linear, alpha_bound_qs),
    ("Dynamical", chi2_dyn_linear, alpha_bound_dyn),
]:
    print(f"\n--- {regime_name} with linear r_int(E) ---")

    # Scan r_foam0 and for each, minimize over (r0, r1)
    def profile_over_linear(rf_val):
        def obj(x):
            r0, r1 = x
            if r0 < 0:
                return 1e10
            return chi2_lin_func([r0, r1, rf_val], E_keV, r_obs, dr_obs, E0_keV)
        res = minimize(obj, [0.10, 0.0], method='Nelder-Mead')
        return res.fun

    chi2_min_lin = profile_over_linear(0.0)
    print(f"  chi2_min (rf=0, linear r_int): {chi2_min_lin:.2f} (dof={len(E_keV)-3})")

    # Find 95% CL upper limit
    target = chi2_min_lin + 4.0
    r_lo, r_hi = 0.0, 0.5
    for _ in range(80):
        r_mid = (r_lo + r_hi) / 2
        if profile_over_linear(r_mid) < target:
            r_lo = r_mid
        else:
            r_hi = r_mid
    r_foam_95_lin = (r_lo + r_hi) / 2

    alpha_95_lin = alpha_func(r_foam_95_lin, D_C, lam0, ell_P)
    print(f"  95% CL upper limit on r_foam,0: {r_foam_95_lin:.4f}")
    print(f"  alpha bound (95% CL, linear r_int): alpha >= {alpha_95_lin:.4f}")


# --- Summary for LaTeX ---
print("\n" + "=" * 70)
print("SUMMARY FOR LATEX")
print("=" * 70)
print(f"ln(D_C/lambda_0) = {np.log(D_C/lam0):.1f}")
print(f"ln(D_C/ell_P) = {np.log(D_C/ell_P):.1f}")
print(f"Single-line bound (R=10): alpha >= {(np.log(D_C/lam0) - np.log(10))/np.log(D_C/ell_P):.2f}")
