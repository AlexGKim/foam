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

# ── Fisher-matrix results from fisher_firas.py (exact) ────────────────────────
sig_A_marg_FIRAS = 1.5116e-17   # m^2  marginal sigma(A_eff)
sig_A_cond_FIRAS = 2.5952e-18   # m^2  conditional sigma(A_eff) [mu,y fixed]
PIXIE_factor     = 1e3          # PIXIE noise floor ~1000x better
sig_A_marg_PIXIE = sig_A_marg_FIRAS / PIXIE_factor
sig_A_cond_PIXIE = sig_A_cond_FIRAS / PIXIE_factor

b_T0             = 1.2120e7    # K / m^2  bias coefficient  Delta_T0 = b_T0 * A_eff
sigma_T0_FIRAS   = 5.70e-4     # K  Fixsen 2009
sigma_T0_PIXIE   = sigma_T0_FIRAS / PIXIE_factor

# ── Alpha grid ────────────────────────────────────────────────────────────────
alpha = np.linspace(0.50, 0.70, 1000)

def A_eff(a):
    """Foam amplitude A_eff(alpha) = D_C^{2(1-a)} * ell_P^{2a}."""
    return D_C**(2*(1 - a)) * ell_P**(2*a)

Ae = A_eff(alpha)

# ── SNR curves ────────────────────────────────────────────────────────────────
snr_FIRAS_marg = Ae / sig_A_marg_FIRAS
snr_FIRAS_cond = Ae / sig_A_cond_FIRAS
snr_PIXIE_marg = Ae / sig_A_marg_PIXIE
snr_PIXIE_cond = Ae / sig_A_cond_PIXIE

# ── Bias curves ────────────────────────────────────────────────────────────────
delta_T0      = b_T0 * Ae                          # K (instrument-independent)
ratio_FIRAS   = delta_T0 / sigma_T0_FIRAS
ratio_PIXIE   = delta_T0 / sigma_T0_PIXIE

# ── Reference alpha values ────────────────────────────────────────────────────
alpha_FIRAS_threshold = 0.57    # marginal SNR ~ 1 (FIRAS)
alpha_BzK             = 0.54    # BzK 4892 spectroscopic lower bound
alpha_GRB_low         = 0.62    # GRB 221009A Ravasio lower bound
alpha_GRB_high        = 0.63    # GRB 221009A Zhang lower bound
alpha_holo            = 2/3     # holographic prediction


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: SNR vs alpha
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))

# --- Four curves ---
ax1.semilogy(alpha, snr_FIRAS_marg, color='C0', lw=2.0, ls='-',
             label=r'FIRAS, marginal (all params free)')
ax1.semilogy(alpha, snr_FIRAS_cond, color='C3', lw=2.0, ls='-',
             label=r'FIRAS, conditional ($\mu,y$ fixed)')
ax1.semilogy(alpha, snr_PIXIE_marg, color='C0', lw=2.0, ls='--',
             label=r'PIXIE, marginal')
ax1.semilogy(alpha, snr_PIXIE_cond, color='C3', lw=2.0, ls='--',
             label=r'PIXIE, conditional ($\mu,y$ fixed)')

# --- Detection threshold ---
ax1.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.7)
ax1.text(0.695, 1.35, r'SNR $= 1$', fontsize=8.5, color='k', alpha=0.8)

# --- Spectroscopic lower bounds ---
ylo, yhi = 1e-6, 3e10
for xv, col, lbl in [
        (alpha_BzK,       '#555555', r'BzK 4892\newline$(\alpha\gtrsim0.54)$'),
        (alpha_GRB_low,   '#888888', r'GRB 221009A\newline$(\alpha\gtrsim0.62)$'),
        (alpha_holo,      '#aaaaaa', r'$\alpha=2/3$'),
]:
    ax1.axvline(xv, color=col, lw=0.8, ls='--', alpha=0.6)

# Label the three vertical lines (placed at mid-height to avoid clipping)
ax1.text(alpha_BzK    + 0.002, 3e3,  r'BzK ($\alpha\!\gtrsim\!0.54$)',
         fontsize=7.5, color='#444444', rotation=90, va='center')
ax1.text(alpha_GRB_low + 0.002, 3e3, r'GRB ($\alpha\!\gtrsim\!0.62$)',
         fontsize=7.5, color='#666666', rotation=90, va='center')
ax1.text(alpha_holo   + 0.002, 3e3,  r'$\alpha=2/3$',
         fontsize=7.5, color='#888888', rotation=90, va='center')

# --- Axes ---
ax1.set_xlim(0.50, 0.70)
ax1.set_ylim(1e-6, 3e10)
ax1.set_xlabel(r'Foam exponent $\alpha$')
ax1.set_ylabel(r'Detection SNR  ($A_\alpha = 1$, $D_C = 13.87\,{\rm Gpc}$)')

# --- Legend: linestyle distinguishes FIRAS/PIXIE, colour distinguishes marg/cond ---
solid_h  = mlines.Line2D([], [], color='k',  lw=2.0, ls='-',  label='FIRAS')
dashed_h = mlines.Line2D([], [], color='k',  lw=2.0, ls='--', label='PIXIE')
blue_h   = mlines.Line2D([], [], color='C0', lw=2.0, ls='-',  label='Marginal (all params free)')
red_h    = mlines.Line2D([], [], color='C3', lw=2.0, ls='-',  label=r'Conditional ($\mu,y$ fixed)')

ax1.legend(handles=[solid_h, dashed_h, blue_h, red_h],
           loc='upper right', framealpha=0.92,
           handlelength=2.2)

ax1.grid(True, which='major', alpha=0.20)
ax1.grid(True, which='minor', alpha=0.07)

plt.tight_layout()
fig1.savefig('cmb_snr_alpha.pdf', bbox_inches='tight')
fig1.savefig('cmb_snr_alpha.png', dpi=160, bbox_inches='tight')
plt.close(fig1)
print('Wrote cmb_snr_alpha.pdf and cmb_snr_alpha.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Bias Delta T_0 / sigma(T_0) vs alpha
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))

# --- Two curves ---
ax2.semilogy(alpha, ratio_FIRAS, color='C0', lw=2.0, ls='-',
             label=r'FIRAS $\;\sigma(T_0)=5.7\times10^{-4}\,{\rm K}$')
ax2.semilogy(alpha, ratio_PIXIE, color='C0', lw=2.0, ls='--',
             label=r'PIXIE $\;\sigma(T_0)=5.7\times10^{-7}\,{\rm K}$')

# --- Threshold line: bias = 1 sigma ---
ax2.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.7)
ax2.text(0.695, 1.35, r'$|\Delta T_0| = \sigma(T_0)$',
         fontsize=8.5, color='k', alpha=0.8, ha='right')

# --- Vertical reference lines ---
for xv, lbl, col in [
        (alpha_BzK,    r'BzK ($\alpha\!\gtrsim\!0.54$)',  '#444444'),
        (alpha_GRB_low, r'GRB ($\alpha\!\gtrsim\!0.62$)', '#666666'),
        (alpha_holo,   r'$\alpha=2/3$',                   '#888888'),
]:
    ax2.axvline(xv, color=col, lw=0.8, ls='--', alpha=0.6)
    ax2.text(xv + 0.002, 1e6, lbl, fontsize=7.5, color=col,
             rotation=90, va='center')

# --- Annotate FIRAS crossover alpha ~ 0.518 ---
alpha_cross_FIRAS = 0.5176
ax2.axvline(alpha_cross_FIRAS, color='C0', lw=0.8, ls=':', alpha=0.5)
ax2.text(alpha_cross_FIRAS + 0.002, 2e-4,
         r'FIRAS bias $= 1\sigma$' + '\n' + r'$\alpha\approx0.518$',
         fontsize=7.5, color='C0', va='bottom')

# --- Axes ---
ax2.set_xlim(0.50, 0.70)
ax2.set_ylim(1e-10, 3e14)
ax2.set_xlabel(r'Foam exponent $\alpha$')
ax2.set_ylabel(r'Temperature bias $|\Delta T_0|/\sigma(T_0)$'
               '\n'
               r'($A_\alpha = 1$, $D_C = 13.87\,{\rm Gpc}$)')

ax2.legend(loc='upper right', framealpha=0.92)
ax2.grid(True, which='major', alpha=0.20)
ax2.grid(True, which='minor', alpha=0.07)

plt.tight_layout()
fig2.savefig('cmb_bias_alpha.pdf', bbox_inches='tight')
fig2.savefig('cmb_bias_alpha.png', dpi=160, bbox_inches='tight')
plt.close(fig2)
print('Wrote cmb_bias_alpha.pdf and cmb_bias_alpha.png')
