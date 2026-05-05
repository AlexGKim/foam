r"""
Figure: foam-induced rms path-length variance sigma_ell(D) for the three
Ng-Perlman exponents, shown against the principal optical sensitivity
thresholds (imaging exclusion via Lieu-Hillman; next-generation HBT
detectability band).

Run:  python make_observability_figure.py
Outputs: observability.pdf  (for \includegraphics in paper.tex)
         observability.png  (for preview)
"""

import numpy as np
import matplotlib.pyplot as plt

# -- Physical constants ------------------------------------------------
ell_P = 1.616e-35              # Planck length [m]
Mpc   = 3.086e22               # 1 Mpc [m]
lam   = 5.0e-7                 # 500 nm [m]
dlam_over_lam = 1.0e-3         # fiducial fractional bandwidth

# -- Distance range ---------------------------------------------------
D_Mpc = np.logspace(-3, 4.5, 500)   # 1 kpc to ~30 Gpc
D     = D_Mpc * Mpc

# -- Ng--Perlman path-length amplitude for the three exponents --------
# sigma_ell(D) = sqrt(A_alpha * D^{2(1-alpha)} * ell_P^{2 alpha}), A_alpha = 1
sig_half = np.sqrt(D * ell_P)              # alpha = 1/2 (random walk)
sig_23   = D**(1./3.) * ell_P**(2./3.)     # alpha = 2/3 (holographic)
sig_one  = ell_P * np.ones_like(D)         # alpha = 1   (conservative)

# -- Lieu-Hillman fringe-erasure threshold ----------------------------
# sigma_phi = 2 pi sigma_ell / lambda = 1  =>  sigma_ell = lambda/(2 pi)
sigma_LH = lam / (2 * np.pi)

# -- HBT detectability band at fiducial bandwidth, fully resolved limit
# delta g^(2) = (4/sqrt(pi)) * sigma_phi * (Delta_lambda/lambda)
#   (Eq. 47 with Lorentzian source, fully resolved Phi_eff -> 0)
# Invert: sigma_phi = (sqrt(pi)/4) * dg2 / (Delta_lambda/lambda)
#         sigma_ell = sigma_phi * lambda / (2 pi)
def sig_ell_for_dg2(dg2):
    sigma_phi = (np.sqrt(np.pi) / 4) * dg2 / dlam_over_lam
    return sigma_phi * lam / (2 * np.pi)

dg2_ceiling = 1e-9    # current/projected HBT systematic ceiling
dg2_floor   = 1e-12   # next-generation modal shot-noise floor
sig_ceiling = sig_ell_for_dg2(dg2_ceiling)
sig_floor   = sig_ell_for_dg2(dg2_floor)

# -- Plot --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 5.2))

# Excluded region above Lieu-Hillman
ax.fill_between(D_Mpc, sigma_LH, 1e-3, color='#d62728', alpha=0.07,
                zorder=0)
# HBT detectability band
ax.fill_between(D_Mpc, sig_floor, sig_ceiling, color='#2ca02c', alpha=0.16,
                zorder=0)
# AGN/quasar D regime
ax.axvspan(1e3, 1e4, color='gray', alpha=0.10, zorder=0)

# Foam model curves
ax.loglog(D_Mpc, sig_half, color='#1f77b4', lw=2.2,
          label=r'$\alpha = 1/2$  (random walk)')
ax.loglog(D_Mpc, sig_23,   color='#2ca02c', lw=2.2,
          label=r'$\alpha = 2/3$  (holographic)')
ax.loglog(D_Mpc, sig_one,  color='#d62728', lw=2.2,
          label=r'$\alpha = 1$  (conservative)')

# Lieu-Hillman line
ax.axhline(sigma_LH, color='k', linestyle='--', lw=1.2, zorder=2)

# -- Annotations -------------------------------------------------------
ax.text(2.5e-3, 1e-9,
        r'Lieu--Hillman fringe-erasure threshold ($\sigma_\phi = 1$, $\lambda = 500$ nm)',
        ha='left', va='bottom', fontsize=9.0)
ax.text(2.5e-3, 3e-5,
        'imaging-excluded\nat optical wavelengths',
        ha='left', va='center', fontsize=8.5, color='#7a1d1d',
        style='italic')
ax.text(2.5e-3, np.sqrt(sig_floor * sig_ceiling),
        'next-generation HBT detectability',
        ha='left', va='center', fontsize=8.5, color='#1d4d1d',
        style='italic')
ax.text(2.5e-3, 8e-18,
        (r'($\delta g^{(2)} \in [10^{-12},10^{-9}]$,'
         r' $\Delta\lambda/\lambda = 10^{-3}$,'
         ' fully resolved)'),
        ha='left', va='top', fontsize=8.5, color='#1d4d1d',
        style='italic')
ax.text(np.sqrt(1e3 * 1e4), 1e-22, 'AGN / quasar', ha='center', va='bottom',
        fontsize=9.5, color='dimgray', rotation=90)

# Off-scale arrow for alpha=1 if y-min cuts it off (it doesn't here, but keep label readable)
# alpha=1 sits at ell_P = 1.6e-35; y-axis spans down that far.

# -- Axes --------------------------------------------------------------
ax.set_xlim(1e-3, 3e4)
ax.set_ylim(1e-37, 1e-3)
ax.set_xlabel(r'Propagation distance $D$ [Mpc]', fontsize=11)
ax.set_ylabel(r'rms foam path-length amplitude $\sigma_\ell(D)$ [m]', fontsize=11)
# Position legend so its top sits at y = 10^-27.
# Axes y-range is 1e-37 to 1e-3 in log; fraction = (-27 - (-37))/(-3 - (-37)) ≈ 0.294.
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.294),
          fontsize=9.5, framealpha=0.95)
ax.grid(True, which='major', alpha=0.25)
ax.grid(True, which='minor', alpha=0.10)

plt.tight_layout()
plt.savefig('observability.pdf')
plt.savefig('observability.png', dpi=160)
print('Wrote observability.pdf and observability.png')
