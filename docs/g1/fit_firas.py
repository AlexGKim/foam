#!/usr/bin/env python3
r"""
Real fit to the public COBE/FIRAS monopole residual spectrum, using the FULL
frequency-frequency covariance.

Replaces the flat-noise Fisher *forecast* (fisher_firas.py) with a generalized-
least-squares fit to the FIRAS calibrated residual spectrum (Fixsen et al. 1996,
Table 4) using the apodization-induced correlation structure from the LOWF
covariance matrix.

Data (NASA LAMBDA):
  data/firas_monopole_spec_v1.txt    cols: nu[cm^-1], monopole[MJy/sr],
                                     residual[kJy/sr], sigma[kJy/sr], galaxy[kJy/sr]
  data/FIRAS_COVARIANCE_MATRIX_LOWF.FITS   43x43 calibrated covariance, (MJy/sr)^2,
                                     grid NU_ZERO=68.02 GHz, DELTA_NU=13.60 GHz.

Covariance used in the fit:  C_ij = R_ij * sigma_i * sigma_j , where R is the
LOWF correlation matrix (dimensionless apodization structure) and sigma is the
published per-channel monopole uncertainty (col 4). This pairs the validated
monopole errors with the real off-diagonal correlations.

Model (linear in the amplitudes):
  residual(nu) = dT0*dB/dT0 + mu*M_mu + y*M_y + g*galaxy + A_eff*foam
with the PHYSICAL no-pedestal foam deficit  foam = -(2 pi nu/c)^2 B_nu(T0).
GLS:  theta = (M^T C^-1 M)^-1 M^T C^-1 r ,  Cov = (M^T C^-1 M)^-1.
"""

import numpy as np
from scipy import constants
from scipy.optimize import brentq
from astropy.io import fits

h, k_B, c = constants.h, constants.k, constants.c
ell_P = 1.616e-35
D_C   = 13.87e3 * 3.0857e22
T0    = 2.725
GHz_per_cm = 29.9792458
kJy   = 1e-23

def A_eff_of_alpha(a, A_alpha=1.0):
    return A_alpha * D_C**(2*(1 - a)) * ell_P**(2*a)

# ── monopole residual spectrum ────────────────────────────────────────────────
d = np.genfromtxt('data/firas_monopole_spec_v1.txt', comments='#')
nu_cm, monopole, resid, sigma, galaxy = d.T
nu = nu_cm * GHz_per_cm * 1e9
n_ch = len(nu)

# ── LOWF covariance -> correlation matrix R, then C = R * (sigma outer sigma) ──
cd = fits.open('data/FIRAS_COVARIANCE_MATRIX_LOWF.FITS')
hdr = cd[0].header
assert hdr['NUM_FREQ'] == n_ch, "covariance channel count mismatch"
nu_cov = (hdr['NU_ZERO'] + np.arange(n_ch) * hdr['DELTA_NU']) * 1e9
assert np.allclose(nu_cov, nu, rtol=2e-3), "covariance frequency grid mismatch"
C_lowf = np.array(cd[1].data['ROW_COVR'])[:, :n_ch]      # (MJy/sr)^2, symmetric
C_lowf = 0.5 * (C_lowf + C_lowf.T)                        # symmetrize (round-off)
dl = np.sqrt(np.diag(C_lowf))
R = C_lowf / np.outer(dl, dl)                            # correlation matrix
C = R * np.outer(sigma, sigma)                           # (kJy/sr)^2
Cdiag = np.diag(sigma**2)
print(f"Loaded {n_ch} FIRAS channels, {nu[0]/1e9:.1f}-{nu[-1]/1e9:.1f} GHz")
print(f"LOWF correlation: max |off-diagonal| = {np.abs(R-np.eye(n_ch)).max():.3f}, "
      f"mean |nearest-neighbour| = {np.mean(np.abs(np.diag(R,1))):.3f}")

# ── templates [kJy/sr] ────────────────────────────────────────────────────────
x  = h * nu / (k_B * T0); ex = np.exp(x)
B  = (2 * h * nu**3 / c**2 / (ex - 1)) / kJy
dB_dT0 = B * x * ex / (ex - 1) / T0
M_mu   = -B * x * ex / (ex - 1)**2
M_y    =  B * x * ex / (ex - 1) * (x*(ex+1)/(ex-1) - 4)
foam   = -(2*np.pi*nu/c)**2 * B

# ── GLS ───────────────────────────────────────────────────────────────────────
def gls(cols, names, cov):
    M = np.array(cols).T
    Cinv = np.linalg.inv(cov)
    A = M.T @ Cinv @ M
    covp = np.linalg.inv(A)
    theta = covp @ (M.T @ Cinv @ resid)
    err = np.sqrt(np.diag(covp))
    chi2 = float((resid - M @ theta) @ Cinv @ (resid - M @ theta))
    return dict(theta=theta, err=err, cov=covp, chi2=chi2,
                dof=n_ch-len(names), names=names)

def report(f):
    for nm, t, e in zip(f['names'], f['theta'], f['err']):
        print(f"    {nm:8s} = {t:+.4e}  +/- {e:.4e}   ({abs(t/e):.2f} sigma)")
    print(f"    chi2/dof = {f['chi2']:.1f}/{f['dof']} = {f['chi2']/f['dof']:.3f}")

def alpha_from_Aeff(A):
    return brentq(lambda a: A_eff_of_alpha(a) - A, 0.40, 0.80)

# ── (1) validation: no foam, FULL covariance ──────────────────────────────────
print("\n" + "="*70 + "\nVALIDATION (no foam), FULL covariance\n" + "="*70)
f0 = gls([dB_dT0, M_mu, M_y, galaxy], ['dT0[K]','mu','y','g_gal'], C)
report(f0)
print(f"\n  Published (Fixsen 1996, 95%):  |mu|<9e-5,  |y|<1.5e-5")
print(f"  This fit  95% (1.96 sigma):    |mu|<{1.96*f0['err'][1]:.1e}, "
      f"|y|<{1.96*f0['err'][2]:.1e}")
# diagonal comparison
f0d = gls([dB_dT0, M_mu, M_y, galaxy], ['dT0[K]','mu','y','g_gal'], Cdiag)
print(f"  diagonal-only 95%:             |mu|<{1.96*f0d['err'][1]:.1e}, "
      f"|y|<{1.96*f0d['err'][2]:.1e}")

# ── (2) foam fit, FULL covariance ─────────────────────────────────────────────
print("\n" + "="*70 + "\nFOAM FIT (physical no-pedestal deficit), FULL covariance\n" + "="*70)
f1 = gls([dB_dT0, M_mu, M_y, galaxy, foam], ['dT0[K]','mu','y','g_gal','A_eff'], C)
report(f1)
A_hat, A_err = f1['theta'][4], f1['err'][4]
A_ul95 = max(A_hat, 0.0) + 1.645 * A_err
print(f"\n  A_eff = ({A_hat:+.3e} +/- {A_err:.3e}) m^2")
print(f"  sigma(A_eff) = {A_err:.3e} m^2   ->  alpha >~ {alpha_from_Aeff(A_err):.3f}"
      f"   [Fisher forecast 2.2e-11, alpha>~0.52]")
print(f"  95% UL  A_eff < {A_ul95:.3e} m^2  ->  alpha >~ {alpha_from_Aeff(A_ul95):.3f}")

# ── (2b) T0/mu/y uncertainty inflation from adding the foam parameter ─────────
print("\n" + "="*70 + "\nUNCERTAINTY INFLATION from the extra (foam) parameter, FULL cov\n" + "="*70)
print(f"  {'param':6s}  {'sigma (no foam)':>16s}  {'sigma (+foam)':>14s}  {'inflation':>10s}")
for nm, i in [('T0', 0), ('mu', 1), ('y', 2)]:
    s0, s1 = f0['err'][i], f1['err'][i]
    print(f"  {nm:6s}  {s0:16.3e}  {s1:14.3e}  {s1/s0:9.1f}x")
print(f"\n  Fixsen 2009 sigma(T0) [calibration] = 5.7e-4 K  "
      f"(cf. foam-marginalized {f1['err'][0]:.1e} K)")
covA = f1['cov']; eA = f1['err']
print("  fit correlations of A_eff with:  "
      + "  ".join(f"{nm}={covA[4,i]/(eA[4]*eA[i]):+.3f}"
                   for nm, i in [('T0',0),('y',2),('mu',1)]))

# ── (2c) T0 shift from UNACCOUNTED foam: bias when A_eff is present but omitted ─
# Inject foam A_eff*foam and fit only (T0,mu,y,gal): bias = (Mf^T C^-1 Mf)^-1 Mf^T C^-1 foam.
print("\n" + "="*70 + "\nT0 SHIFT from UNACCOUNTED foam (fit omits A_eff), FULL cov\n" + "="*70)
Mf = np.array([dB_dT0, M_mu, M_y, galaxy]).T
b_bias = np.linalg.solve(Mf.T @ np.linalg.inv(C) @ Mf, Mf.T @ np.linalg.inv(C) @ foam)
bT0 = b_bias[0]                                  # K / m^2:  Delta_T0 = bT0 * A_eff
print(f"  bias coefficient  b_T0 = {bT0:+.3e} K/m^2   (Delta_T0 = b_T0 * A_eff)")
print(f"  {'alpha':>7s}  {'A_eff(A_alpha=1) [m^2]':>22s}  {'|Delta_T0| [K]':>14s}  "
      f"{'/sigma(T0)=5.7e-4':>17s}")
for a in [0.50, 0.52, 2/3, 1.0]:
    A = A_eff_of_alpha(a); print(f"  {a:7.3f}  {A:22.3e}  {abs(bT0*A):14.3e}  "
                                 f"{abs(bT0*A)/5.7e-4:17.2e}")
print(f"  => |Delta_T0|=sigma(T0) only at A_eff={5.7e-4/abs(bT0):.2e} m^2 "
      f"(alpha~{alpha_from_Aeff(5.7e-4/abs(bT0)):.3f}, foam already detectable)")

# ── (3) comparison table: diagonal vs full; with/without galaxy ───────────────
print("\n" + "="*70 + "\nCOMPARISON: sigma(A_eff) and alpha threshold\n" + "="*70)
for lbl, cols, names in [
    ('full cov,  +galaxy', [dB_dT0,M_mu,M_y,galaxy,foam], ['dT0','mu','y','g','A']),
    ('full cov,  no galaxy',[dB_dT0,M_mu,M_y,foam],        ['dT0','mu','y','A']),
    ('diag only, +galaxy', [dB_dT0,M_mu,M_y,galaxy,foam], ['dT0','mu','y','g','A']),
]:
    f = gls(cols, names, C if 'full' in lbl else Cdiag)
    sA = f['err'][-1]
    print(f"  {lbl:22s}: sigma(A_eff)={sA:.3e} m^2  ->  alpha >~ {alpha_from_Aeff(sA):.3f}")
