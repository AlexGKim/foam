#!/usr/bin/env python3
r"""
CMB Fisher-matrix figures for the FIRAS/PIXIE foam analysis.
Kim, Nugent & Wang — g1.tex §V.C

Produces two PDF/PNG figures:

  cmb_snr_alpha.pdf / .png
      SNR for detecting the foam amplitude A_eff as a function of
      foam exponent alpha, for four cases:
        FIRAS marginal  (solid blue)   — all four params free
        FIRAS conditional (solid red)  — mu, y held fixed
        PIXIE marginal  (dashed blue)
        PIXIE conditional (dashed red)

  cmb_bias_alpha.pdf / .png
      Temperature bias Delta T_0 / sigma(T_0) as a function of alpha
      when foam is present but omitted from the spectral fit, for:
        FIRAS (solid blue)   sigma(T_0) = 5.7e-4 K  [Fixsen 2009]
        PIXIE (dashed blue)  sigma(T_0) = 5.7e-7 K  (~3 orders of magnitude better)

Physical inputs (from fisher_firas.py):
  D_C  = 13.87 Gpc (comoving distance to last scattering)
  ell_P = 1.616e-35 m
  A_eff(alpha) = D_C^{2(1-alpha)} * ell_P^{2*alpha}  [A_alpha = 1]

  FIRAS: sigma(A_eff)_marg = 1.5116e-17 m^2
         sigma(A_eff)_cond = 2.5952e-18 m^2  (mu,y fixed)
         sigma(T_0)        = 5.70e-4 K        [Fixsen 2009]
         b_T0              = 1.2120e+7 K/m^2  (bias coefficient)

  PIXIE: noise floor ~1000x lower -> SNR ~1000x higher
         sigma(T_0)_PIXIE  = 5.70e-7 K

Run:  python make_cmb_fisher_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Use LaTeX rendering ────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amssymb}',
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# ── Physical / cosmological constants ─────────────────────────────────────────
ell_P  = 1.616e-35          # Planck length [m]
D_C    = 13.87e3 * 3.0857e22  # 13.87 Gpc [m]

# Comoving-foam model: the (1+z')^2 path weighting enters as an alpha- and
# nu-independent enhancement (lambda_obs/lambda_eff)^2 = <(1+z')^2>_{D_C} ~ 1.2e4
# of the observed amplitude over the bare Ng-Perlman variance, shifting every
# SNR=1 threshold by Delta_alpha = ln(ENH)/[2 ln(D_C/ell_P)] ~ +0.034.
# Value: <(1+z')^2>_{D_C} = int (1+z)^2 dD_C / int dD_C, Planck 2018 to z_LS=1089.
ENH    = 1.365e4           # <(1+z')^2>_{D_C} for the CMB path

# ── Fisher-matrix results from fisher_firas.py (correctly normalized) ─────────
# Per-channel diagonal Fisher F = sum_ch dI_i dI_j / sigma^2 (no bandwidth factor).
# PHYSICAL no-pedestal template (the headline case): pure deficit -(2*pi*nu/c)^2 B.
sig_A_marg_FIRAS = 2.2222e-11   # m^2  marginal sigma(A_eff), no-pedestal
sig_A_cond_FIRAS = 8.2089e-13   # m^2  conditional sigma(A_eff) [mu,y fixed], no-pedestal

# Next-gen instruments scale by their forecast mu-sensitivity relative to FIRAS:
#   factor = sigma(mu)_FIRAS / sigma(mu)_inst  (SNR ~ 1/sigma(A_eff) ~ 1/sigma(mu))
# FIRAS ~5.5e-5; PIXIE 3.0e-8; SuperPIXIE 7.7e-9; Voyage 2050 2.2e-9 [Chluba+2021]
PIXIE_factor  = 5.5e-5 / 3.0e-8   # ~1833
SPIXIE_factor = 5.5e-5 / 7.7e-9   # ~7143
VOY_factor    = 5.5e-5 / 2.2e-9   # ~25000

sig_A_marg_PIXIE  = sig_A_marg_FIRAS / PIXIE_factor
sig_A_cond_PIXIE  = sig_A_cond_FIRAS / PIXIE_factor
sig_A_marg_SPIXIE = sig_A_marg_FIRAS / SPIXIE_factor
sig_A_cond_SPIXIE = sig_A_cond_FIRAS / SPIXIE_factor
sig_A_marg_VOY    = sig_A_marg_FIRAS / VOY_factor
sig_A_cond_VOY    = sig_A_cond_FIRAS / VOY_factor

# ── Alpha grid ────────────────────────────────────────────────────────────────
alpha = np.linspace(0.50, 0.70, 1000)

def A_eff(a):
    """Comoving-foam amplitude A_eff(alpha) = ENH * D_C^{2(1-a)} * ell_P^{2a}."""
    return ENH * D_C**(2*(1 - a)) * ell_P**(2*a)

Ae = A_eff(alpha)

# ── SNR curves ────────────────────────────────────────────────────────────────
snr_FIRAS_marg  = Ae / sig_A_marg_FIRAS
snr_FIRAS_cond  = Ae / sig_A_cond_FIRAS
snr_PIXIE_marg  = Ae / sig_A_marg_PIXIE
snr_PIXIE_cond  = Ae / sig_A_cond_PIXIE
snr_SPIXIE_marg = Ae / sig_A_marg_SPIXIE
snr_SPIXIE_cond = Ae / sig_A_cond_SPIXIE
snr_VOY_marg    = Ae / sig_A_marg_VOY
snr_VOY_cond    = Ae / sig_A_cond_VOY

# ── Reference alpha values ────────────────────────────────────────────────────
alpha_FIRAS_threshold = 0.55    # marginal SNR ~ 1 (FIRAS, comoving-foam)
alpha_GRB             = 0.63    # GRB 221009A Zhang lower bound (headline, Eq. 24)
alpha_holo            = 2/3     # holographic prediction


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: SNR vs alpha
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))

# --- Marginal curves: one per instrument (blue, linestyle = instrument) ---
ax1.semilogy(alpha, snr_FIRAS_marg,  color='C0', lw=2.0, ls='-',
             label=r'FIRAS, marginal (all params free)')
ax1.semilogy(alpha, snr_PIXIE_marg,  color='C0', lw=2.0, ls='--',
             label=r'PIXIE, marginal')
ax1.semilogy(alpha, snr_SPIXIE_marg, color='C0', lw=1.8, ls='-.',
             label=r'SuperPIXIE, marginal')
ax1.semilogy(alpha, snr_VOY_marg,    color='C0', lw=1.8, ls=':',
             label=r'Voyage 2050, marginal')
# --- Conditional curves (mu,y fixed), one per instrument, showing the
#     foam-mu degeneracy penalty that marginalization incurs ---
ax1.semilogy(alpha, snr_FIRAS_cond,  color='C3', lw=2.0, ls='-',
             label=r'FIRAS, conditional ($\mu,y$ fixed)')
ax1.semilogy(alpha, snr_PIXIE_cond,  color='C3', lw=2.0, ls='--',
             label=r'PIXIE, conditional ($\mu,y$ fixed)')
ax1.semilogy(alpha, snr_SPIXIE_cond, color='C3', lw=1.8, ls='-.',
             label=r'SuperPIXIE, conditional ($\mu,y$ fixed)')
ax1.semilogy(alpha, snr_VOY_cond,    color='C3', lw=1.8, ls=':',
             label=r'Voyage 2050, conditional ($\mu,y$ fixed)')

# --- Detection threshold ---
ax1.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.7)
ax1.text(0.695, 1.35, r'SNR $= 1$', fontsize=8.5, color='k', alpha=0.8)

# --- Spectroscopic lower bounds ---
ylo, yhi = 1e-6, 3e10
for xv, col, lbl in [
        (alpha_GRB,       '#888888', r'GRB 221009A\newline$(\alpha\gtrsim0.63)$'),
        (alpha_holo,      '#aaaaaa', r'$\alpha=2/3$'),
]:
    ax1.axvline(xv, color=col, lw=0.8, ls='--', alpha=0.6)

# Label the two vertical lines (placed at mid-height to avoid clipping)
ax1.text(alpha_GRB + 0.002, 3e3, r'GRB ($\alpha\!\gtrsim\!0.63$)',
         fontsize=7.5, color='#666666', rotation=90, va='center')
ax1.text(alpha_holo   + 0.002, 3e3,  r'$\alpha=2/3$',
         fontsize=7.5, color='#888888', rotation=90, va='center')

# --- Axes ---
ax1.set_xlim(0.50, 0.70)
ax1.set_ylim(1e-6, 3e10)
ax1.set_xlabel(r'Foam exponent $\alpha$')
ax1.set_ylabel(r'Detection SNR  ($A_\alpha = 1$, $D_C = 13.87\,{\rm Gpc}$)')

# --- Legend: linestyle distinguishes instrument, colour distinguishes marg/cond ---
firas_h  = mlines.Line2D([], [], color='k',  lw=2.0, ls='-',  label='FIRAS')
pixie_h  = mlines.Line2D([], [], color='k',  lw=2.0, ls='--', label='PIXIE')
spixie_h = mlines.Line2D([], [], color='k',  lw=1.8, ls='-.', label='SuperPIXIE')
voy_h    = mlines.Line2D([], [], color='k',  lw=1.8, ls=':',  label='Voyage 2050')
blue_h   = mlines.Line2D([], [], color='C0', lw=2.0, ls='-',  label='Marginal (all params free)')
red_h    = mlines.Line2D([], [], color='C3', lw=2.0, ls='-',  label=r'Conditional ($\mu,y$ fixed)')

ax1.legend(handles=[firas_h, pixie_h, spixie_h, voy_h, blue_h, red_h],
           loc='upper right', framealpha=0.92,
           handlelength=2.6, fontsize=8.5)

ax1.grid(True, which='major', alpha=0.20)
ax1.grid(True, which='minor', alpha=0.07)

plt.tight_layout()
fig1.savefig('cmb_snr_alpha.pdf', bbox_inches='tight')
fig1.savefig('cmb_snr_alpha.png', dpi=160, bbox_inches='tight')
plt.close(fig1)
print('Wrote cmb_snr_alpha.pdf and cmb_snr_alpha.png')
# (The former Figure 2, the T0-bias-vs-alpha plot, was removed: the T0 result is
#  now the fit-based uncertainty-inflation factor reported in fit_firas.py.)
