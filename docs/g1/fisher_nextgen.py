#!/usr/bin/env python3
"""Next-generation CMB spectrometer foam forecast: SuperPIXIE & Voyage 2050.

Extends fisher_firas.py to the wide bands (10 GHz - few THz) of the
SuperPIXIE / Voyage 2050 concepts, to test whether the broadened frequency
coverage RELAXES the foam-mu (and foam-T0) degeneracy that caps the FIRAS
constraint, beyond the naive logarithmic sensitivity scaling.

Strategy
--------
The foam threshold solves  A_eff_of_alpha(alpha) = sigma(A_eff), with

    sigma(A_eff) = sigma(mu)_published  x  R_band,
    R_band       = [sigma(A_eff) / sigma(mu)]  (normalization-independent).

Two levers move the threshold relative to FIRAS:
  (1) sigma(mu) depth   -> published forecasts (band+noise+foregrounds).
  (2) R_band            -> set by band SHAPE only; tests degeneracy relaxation.

Both R_band and the correlation coefficients rho(A_eff, mu), rho(A_eff, T0)
are independent of the overall white-noise level (it cancels), so they are
computed with flat per-channel noise and are robust to the absolute
calibration.  The threshold is then quoted RELATIVE to the paper's FIRAS
anchor alpha<=0.56, isolating the extra gain from R-reduction:

    Delta_alpha = ln[ (sigma_mu_FIRAS/sigma_mu_exp) (R_FIRAS/R_exp) ] / (2 ln(D_C/ell_P))

Foam templates (linearized, C_alpha->0):
  no-pedestal (physical):  dI/dA = -(2 pi nu/c)^2 B_nu(T0)
  adversarial pedestal:    dI/dA = -(2 pi nu/c)^2 B_nu + K_fast * P_foam
                           K_fast = 1/(nu ln(nu_max/nu_min))  (per band)
"""

import numpy as np
from scipy import constants
from scipy.optimize import brentq

h, k_B, c = constants.h, constants.k, constants.c
ell_P = 1.616e-35                      # m
D_C   = 13.87e3 * 3.0857e22           # 13.87 Gpc in m
T0    = 2.7255                        # K
MJy   = 1e-20                         # W m^-2 Hz^-1 sr^-1
LOG_DC = np.log(D_C / ell_P)          # ~141 ; 2*LOG_DC ~ 282

def A_eff_of_alpha(alpha, A_alpha=1.0):
    return A_alpha * D_C**(2*(1 - alpha)) * ell_P**(2*alpha)

def templates(nu, pedestal):
    """Return derivative templates (dI/dA, dI/dT0, dI/dmu, dI/dy) on grid nu."""
    x  = h * nu / (k_B * T0)
    ex = np.exp(x)
    B  = 2 * h * nu**3 / c**2 / (ex - 1)
    k_nu = (2 * np.pi * nu / c)**2                # sigma_phi^2 = k_nu * A_eff
    dI_dT  = B * x * ex / (ex - 1) / T0
    dI_dmu = -B * x * ex / (ex - 1)**2
    dI_dy  = B * x * ex / (ex - 1) * (x * (ex + 1) / (ex - 1) - 4)
    if pedestal:
        dln = np.log(nu[-1] / nu[0])
        K_fast = 1.0 / (nu * dln)
        P_foam = np.trapezoid(k_nu * B, nu)
        dI_dA = -k_nu * B + K_fast * P_foam
    else:
        dI_dA = -k_nu * B
    return np.array([dI_dA, dI_dT, dI_dmu, dI_dy])

def fisher(nu, pedestal, sigma_noise=1e-22):
    """Diagonal-noise Fisher (code convention: F = sum d_i d_j dnu / sigma^2)."""
    derivs = templates(nu, pedestal)
    dnu = np.gradient(nu)
    F = np.einsum('ik,jk,k->ij', derivs, derivs, dnu / sigma_noise**2)
    Cov = np.linalg.inv(F)
    sig = np.sqrt(np.diag(Cov))
    rho = Cov / np.outer(sig, sig)
    return sig, rho

NAMES = ['A_eff', 'T_0', 'mu', 'y']

# Band definitions (GHz -> Hz); fine uniform-in-nu grid for the continuum Fisher
BANDS = {
    'FIRAS            (60-600 GHz)' : (60e9,   600e9),
    'SuperPIXIE/Voyage(10-3000 GHz)': (10e9,  3000e9),
    'PIXIE/Kogut      (10-6000 GHz)': (10e9,  6000e9),
}

# Published 1-sigma mu sensitivities (band+noise+foregrounds) used as depth anchors
SIGMA_MU_PUB = {
    'FIRAS'      : 5.5e-5,   # ~ 9e-5 (95%) / 1.64
    'PIXIE'      : 3.0e-8,   # Chluba+2021
    'SuperPIXIE' : 7.7e-9,   # Chluba+2021
    'Voyage2050' : 2.2e-9,   # Chluba+2021 (scaled SuperPIXIE x5)
}

print("="*88)
print("BAND-SHAPE DEGENERACY TEST  (normalization-independent: flat noise)")
print("="*88)
results = {}
for tmpl, ped in [('NO-PEDESTAL (physical)', False), ('ADVERSARIAL pedestal', True)]:
    print(f"\n--- foam template: {tmpl} ---")
    print(f"  {'band':32s}  {'rho(A,mu)':>10s}  {'rho(A,T0)':>10s}  {'rho(A,y)':>9s}  "
          f"{'R=sig(A)/sig(mu) [m^2]':>22s}")
    for label, (lo, hi) in BANDS.items():
        nu = np.linspace(lo, hi, 4000)
        sig, rho = fisher(nu, ped)
        R = sig[0] / sig[2]
        results[(tmpl, label)] = (rho[0,2], rho[0,1], rho[0,3], R)
        print(f"  {label:32s}  {rho[0,2]:10.4f}  {rho[0,1]:10.4f}  {rho[0,3]:9.4f}  {R:22.4e}")

# ---------------------------------------------------------------------------
# Threshold relative to FIRAS anchor alpha<=0.56, isolating the R-reduction gain
# ---------------------------------------------------------------------------
print("\n" + "="*88)
print("FOAM THRESHOLD  (anchored to paper FIRAS alpha<=0.56; physical no-pedestal)")
print("="*88)
tmpl = 'NO-PEDESTAL (physical)'
R_firas = results[(tmpl, 'FIRAS            (60-600 GHz)')][3]
alpha_anchor = 0.52   # corrected FIRAS no-pedestal threshold (normalization fixed)

def alpha_threshold(sigma_A):
    return brentq(lambda a: A_eff_of_alpha(a) - sigma_A, 0.45, 0.75)

# sigma(A_eff) for FIRAS implied by the anchor
sigA_firas = A_eff_of_alpha(alpha_anchor)

print(f"\nFIRAS anchor: alpha={alpha_anchor:.3f} -> sigma(A_eff)={sigA_firas:.3e} m^2,"
      f"  R_FIRAS={R_firas:.3e} m^2")
print(f"Implied FIRAS sigma(mu) in paper normalization = sigA/R = "
      f"{sigA_firas/R_firas:.3e}  (cf. real FIRAS ~{SIGMA_MU_PUB['FIRAS']:.1e})\n")

print(f"  {'experiment':14s}  {'band':16s}  {'sig(mu)_pub':>11s}  {'R_band[m^2]':>12s}  "
      f"{'R_FIR/R':>8s}  {'sigma(A)[m^2]':>13s}  {'alpha_thr':>9s}  {'Dalpha_R':>9s}")
EXP = [
    ('PIXIE',      'PIXIE/Kogut      (10-6000 GHz)'),
    ('SuperPIXIE', 'SuperPIXIE/Voyage(10-3000 GHz)'),
    ('Voyage2050', 'SuperPIXIE/Voyage(10-3000 GHz)'),
]
# log-only reference (R held at FIRAS value): pure sensitivity scaling
print("\n  [log-only, R fixed at FIRAS -> pure sensitivity scaling]")
for name, band in EXP:
    ratio_mu = SIGMA_MU_PUB['FIRAS'] / SIGMA_MU_PUB[name]
    dalpha = np.log(ratio_mu) / (2*LOG_DC)
    print(f"    {name:14s}  alpha_thr = {alpha_anchor + dalpha:.3f}   (Dalpha={dalpha:+.3f})")

print("\n  [full: sensitivity x band-shape R-reduction]")
for name, band in EXP:
    R_band = results[(tmpl, band)][3]
    ratio_mu = SIGMA_MU_PUB['FIRAS'] / SIGMA_MU_PUB[name]
    ratio_R  = R_firas / R_band
    # sigma(A_eff)_exp = sigma(A)_FIRAS / (ratio_mu * ratio_R)
    sigA = sigA_firas / (ratio_mu * ratio_R)
    a_thr = alpha_threshold(sigA)
    dalpha_R = np.log(ratio_R) / (2*LOG_DC)
    print(f"  {name:14s}  {band.split('(')[1][:-1]:16s}  {SIGMA_MU_PUB[name]:11.1e}  "
          f"{R_band:12.3e}  {ratio_R:8.2f}  {sigA:13.3e}  {a_thr:9.3f}  {dalpha_R:+9.3f}")

print("\nNote: rho and R use flat per-channel noise (cancels in both). Threshold is")
print("anchored to the paper's FIRAS alpha<=0.56; 'Dalpha_R' is the EXTRA shift from")
print("band-shape degeneracy reduction beyond the pure-sensitivity (log-only) scaling.")
