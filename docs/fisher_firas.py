#!/usr/bin/env python3
"""Fisher matrix analysis for FIRAS CMB foam constraint.

Scope: determine how well A_eff (the combined foam amplitude) can be constrained
by FIRAS, and what degeneracies with standard spectral distortions (mu, y)
dominate the error budget.

Model parameters: theta = (A_eff, T_0, mu, y)
  A_eff = A_alpha * D_C^{2(1-alpha)} * ell_P^{2*alpha}
  sigma_phi^2(nu) = (2*pi*nu/c)^2 * A_eff

Foam spectral template (fully dynamical C_alpha -> 0 limit, linearized):
  Delta_I(nu) = -sigma_phi^2(nu) * B_nu(T_0)
              + K_nu^fast * int_band sigma_phi^2(nu') * B_nu'(T_0) d nu'

  K_nu^fast = 1 / (nu * ln(nu_max/nu_min))   [placeholder — uniform in ln nu]

Standard distortion templates (Fixsen 1996, Chluba & Sunyaev 2012):
  Delta_I^mu = -mu * B_nu * x*e^x / (e^x - 1)^2
  Delta_I^y  =  y  * B_nu * x*e^x / (e^x - 1) * [x*(e^x+1)/(e^x-1) - 4]

FIRAS noise: diagonal approximation, sigma_nu ~ 0.01 MJy/sr per channel
(Fixsen 1996 Table 4; actual covariance is correlated but this gives
order-of-magnitude correct Fisher constraints).

References:
  Fixsen et al. 1996, ApJ 473, 576
  Chluba & Sunyaev 2012, MNRAS 419, 1294
"""

import numpy as np
from scipy import constants

# ── Physical constants ────────────────────────────────────────────────────────
h    = constants.h        # J s
k_B  = constants.k        # J/K
c    = constants.c        # m/s
ell_P = 1.616e-35         # Planck length, m

# ── Cosmology ─────────────────────────────────────────────────────────────────
# Planck 2018 flat ΛCDM: H_0=67.36 km/s/Mpc, Ω_m=0.315
# Comoving distance to last scattering z_CMB≈1089
D_C = 13.87e3 * 3.0857e22  # 13.87 Gpc in metres

# ── Fiducial CMB temperature ──────────────────────────────────────────────────
T0 = 2.7255  # K (Fixsen 2009)

# ── FIRAS band ────────────────────────────────────────────────────────────────
nu_min =  60e9   #  60 GHz  ( 2 cm^{-1})
nu_max = 600e9   # 600 GHz  (20 cm^{-1})
n_ch   = 43
nu = np.linspace(nu_min, nu_max, n_ch)  # uniform channels
dnu = nu[1] - nu[0]

# ── Conversions ───────────────────────────────────────────────────────────────
MJy_per_sr = 1e-20   # 1 MJy/sr = 1e-20 W m^{-2} Hz^{-1} sr^{-1}

# ── Planck function and reduced variables ─────────────────────────────────────
x = h * nu / (k_B * T0)                          # dimensionless
ex = np.exp(x)
B = 2 * h * nu**3 / c**2 / (ex - 1)              # W m^{-2} Hz^{-1} sr^{-1}
B_MJy = B / MJy_per_sr

# ── Foam phase variance ───────────────────────────────────────────────────────
k_nu = (2 * np.pi * nu / c)**2                    # m^{-2}; sigma_phi^2 = k_nu * A_eff

def A_eff_of_alpha(alpha, A_alpha=1.0):
    """Combined foam amplitude for given exponent alpha (Ng-Perlman A_alpha=1)."""
    return A_alpha * D_C**(2*(1 - alpha)) * ell_P**(2*alpha)

# ── Pedestal kernel (placeholder: uniform in ln nu) ───────────────────────────
dln_nu = np.log(nu_max / nu_min)
K_fast = 1.0 / (nu * dln_nu)                      # Hz^{-1}; integral over nu gives 1

# ── Template derivatives at fiducial (A_eff=0, T0, mu=0, y=0) ────────────────
# d(I) / d(A_eff)
pedestal_integrand = k_nu * B                       # W m^{-4} sr^{-1}
pedestal_integral  = np.trapz(pedestal_integrand, nu)  # W m^{-4} Hz sr^{-1}  (NOT right dim — cancel below)
# Actually: int k_nu' B_nu' dnu' has units [m^{-2}] * [W m^{-2} Hz^{-1} sr^{-1}] * [Hz] = W m^{-4} sr^{-1}
# K_fast * pedestal_integral has units [Hz^{-1}] * [W m^{-4} sr^{-1}] = W m^{-4} Hz^{-1} sr^{-1}
# That is NOT the same as B (W m^{-2} Hz^{-1} sr^{-1}) because k_nu has units m^{-2}:
# d(I)/d(A_eff) has units of W m^{-4} Hz^{-1} sr^{-1}  (times A_eff [m^2] gives W m^{-2} Hz^{-1} sr^{-1})

dI_dA   = -k_nu * B + K_fast * pedestal_integral   # W m^{-4} Hz^{-1} sr^{-1}

# d(I) / d(T_0)
dI_dT   = B * x * ex / (ex - 1) / T0               # W m^{-2} Hz^{-1} sr^{-1} K^{-1}

# d(I) / d(mu)
dI_dmu  = -B * x * ex / (ex - 1)**2                # W m^{-2} Hz^{-1} sr^{-1}

# d(I) / d(y)
dI_dy   = B * x * ex / (ex - 1) * (x * (ex + 1) / (ex - 1) - 4)  # same units

# ── FIRAS noise covariance (diagonal approximation) ───────────────────────────
# Fixsen 1996 Table 4: noise ≈ 0.5–3 × 10^{-22} W m^{-2} Hz^{-1} sr^{-1}
# across the band; use sigma = 0.01 MJy/sr = 1e-22 W/m^2/Hz/sr as representative.
sigma_noise = 0.01 * MJy_per_sr   # W m^{-2} Hz^{-1} sr^{-1}

# Fisher matrix F_ij = Σ_nu (dI_i / sigma)^2 [uniform diagonal noise]
derivs = np.array([dI_dA, dI_dT, dI_dmu, dI_dy])  # (4, n_ch)
names  = ['A_eff', 'T_0',  'mu',   'y']

F = np.einsum('ik,jk->ij', derivs, derivs) * (dnu / sigma_noise**2)

# Parameter covariance
Cov = np.linalg.inv(F)
sig = np.sqrt(np.diag(Cov))
rho = Cov / np.outer(sig, sig)

# ── Results ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("FIRAS FOAM FISHER MATRIX ANALYSIS")
print("Diagonal noise approx  sigma = 0.01 MJy/sr per channel  (43 channels)")
print("=" * 72)

print(f"\nFiducial:  T_0 = {T0} K,  A_eff = mu = y = 0")
print(f"D_C = {D_C/3.0857e25:.2f} Gpc,  ell_P = {ell_P:.3e} m\n")

# Sigma_phi^2 spot-checks
print("sigma_phi^2 at benchmark frequencies (alpha-dependent):")
print(f"  {'alpha':>6s}  {'A_eff [m^2]':>14s}  {'nu=60GHz':>10s}  {'nu=150GHz':>10s}  {'nu=600GHz':>10s}")
for alpha in [0.50, 0.55, 0.59, 2/3]:
    Ae = A_eff_of_alpha(alpha)
    s60  = k_nu[ 0] * Ae
    s150 = (2*np.pi*150e9/c)**2 * Ae
    s600 = k_nu[-1] * Ae
    print(f"  {alpha:6.3f}  {Ae:14.3e}  {s60:10.3e}  {s150:10.3e}  {s600:10.3e}")

print()
print("Marginal 1-sigma uncertainties (marginalizing over all other params):")
for i, name in enumerate(names):
    print(f"  sigma({name:6s}) = {sig[i]:.4e}")

print()
print("Conditional sigma(A_eff) [mu,y held fixed]:")
idx = [0, 1]   # A_eff, T_0 only
Cov_cond = np.linalg.inv(F[np.ix_(idx, idx)])
print(f"  sigma(A_eff)|_{{mu,y fixed}} = {np.sqrt(Cov_cond[0,0]):.4e}")

print()
print("Correlation matrix  rho_ij:")
header = "".join(f"  {n:>10s}" for n in names)
print(f"  {'':10s}{header}")
for i, name in enumerate(names):
    row = "".join(f"  {rho[i,j]:10.4f}" for j in range(4))
    print(f"  {name:10s}{row}")

print()
print("Detection SNR for foam at various alpha (A_alpha=1):")
print(f"  {'alpha':>6s}  {'A_eff [m^2]':>14s}  {'SNR (marginal)':>16s}  {'SNR (mu,y fixed)':>18s}")
for alpha in np.arange(0.50, 0.66, 0.01):
    Ae = A_eff_of_alpha(alpha)
    snr_marg = Ae / sig[0]
    snr_cond = Ae / np.sqrt(Cov_cond[0, 0])
    print(f"  {alpha:6.3f}  {Ae:14.3e}  {snr_marg:16.2f}  {snr_cond:18.2f}")

print()
print("Conversion  sigma(A_eff) -> sigma(A_alpha)  at fixed alpha:")
print("  sigma(A_alpha) = sigma(A_eff) / [D_C^{2(1-alpha)} * ell_P^{2*alpha}]")
print(f"  {'alpha':>6s}  {'A_eff [m^2]':>14s}  {'sigma(A_alpha):>14s}'}")
for alpha in [0.50, 0.55]:
    Ae  = A_eff_of_alpha(alpha)
    denom = D_C**(2*(1-alpha)) * ell_P**(2*alpha)
    s_Aalpha = sig[0] / denom
    print(f"  {alpha:6.3f}  {Ae:14.3e}  {s_Aalpha:14.3e}")

print()
print("Pedestal flux fraction (diagnostic — should integrate to ~0 for photon conservation):")
foam_deficit  = np.trapz(k_nu * B, nu)          # int sigma_phi^2 * B dnu  (A_eff=1 normalised)
pedestal_flux = pedestal_integral * dln_nu       # K_fast * integral * nu dln_nu
print(f"  int k_nu B_nu dnu          = {foam_deficit:.4e}  W m^{{-4}} sr^{{-1}}")
print(f"  int K_fast * pedestal * nu d ln_nu (check) = {np.trapz(K_fast * pedestal_integral * nu, nu):.4e}")
net = np.trapz(dI_dA, nu)
print(f"  int (dI/dA_eff) dnu        = {net:.4e}  (should be ~0 if photon-number-conserving)")

print()
print("Peak CMB intensity and foam signal amplitude (alpha=0.5, A_alpha=1):")
Ae_half = A_eff_of_alpha(0.5)
peak_B = np.max(B_MJy)
foam_amp_150 = (2*np.pi*150e9/c)**2 * Ae_half * max(B_MJy[nu < 151e9])
print(f"  Peak B_nu(T_0)             = {peak_B:.1f} MJy/sr  at nu ~ 160 GHz")
print(f"  |Delta_I_foam|(150 GHz)    = {foam_amp_150:.2f} MJy/sr  (sigma_phi^2 x B)")
print(f"  FIRAS noise sigma_nu       = {sigma_noise/MJy_per_sr:.4f} MJy/sr")
print(f"  SNR per channel (150GHz)   = {foam_amp_150 / (sigma_noise/MJy_per_sr):.1f}")
