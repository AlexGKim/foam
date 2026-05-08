r"""
Figure: foam-induced rms path-length amplitude sigma_ell(D) for the three
Ng-Perlman exponents, against the principal optical sensitivity thresholds
(imaging exclusion via Lieu-Hillman; HBT Siegert-violation accessible band).

Run:  python make_observability_figure.py
Outputs: observability.pdf  (for \includegraphics in paper.tex)
         observability.png  (for preview)
"""

import numpy as np
import matplotlib.pyplot as plt

# ===========================================================
# Physical constants
# ===========================================================
ell_P   = 1.616e-35   # Planck length [m]
Mpc     = 3.086e22    # 1 Mpc [m]
lam_opt = 5.0e-7      # 500 nm
lam_uv  = 2.5e-7      # 250 nm

# ===========================================================
# Distance range and Ng--Perlman amplitudes (A_alpha = 1)
# ===========================================================
D_Mpc = np.logspace(-3, 4.5, 600)
D     = D_Mpc * Mpc
sig_half = np.sqrt(D * ell_P)              # alpha = 1/2 (random walk)
sig_23   = D**(1./3.) * ell_P**(2./3.)     # alpha = 2/3 (holographic)

# ===========================================================
# Sensitivity thresholds
# ===========================================================
sigma_LH     = lam_opt / (2 * np.pi)              # Lieu--Hillman, sigma_phi = 1
sig_ground   = 0.8  * lam_opt / (2 * np.pi)       # ground HBT, sigma_phi = 0.8
sig_space_uv = 0.15 * lam_uv  / (2 * np.pi)       # space UV, sigma_phi = 0.15

# ===========================================================
# Color palette (chosen to be distinct from band shadings)
# ===========================================================
C_HALF   = '#08306b'      # deep navy blue   (alpha = 1/2)
C_TWOTHR = '#ad3803'      # burnt orange     (alpha = 2/3)
C_EXCL   = '#d62728'      # red   for imaging-excluded shading
C_HBT    = '#2ca02c'      # green for HBT-accessible band
C_HBT_DK = '#0e4d0e'      # darker green for HBT text/edges
C_GRAY   = '#555555'

# ===========================================================
# Plot
# ===========================================================
fig, ax = plt.subplots(figsize=(7.2, 5.0))

xmin, xmax = 1e-3, 3e4
ymin, ymax = 1e-18, 1e-3

# ----- shaded regions (background, low z-order) ----------------
# Imaging-excluded: above the Lieu-Hillman line
ax.fill_between(D_Mpc, sigma_LH, ymax, color=C_EXCL, alpha=0.10, zorder=0)
# HBT Siegert-violation accessible: between space-UV and ground thresholds
ax.fill_between(D_Mpc, sig_space_uv, sig_ground,
                color=C_HBT, alpha=0.20, zorder=0)
# Galactic transient distance band (eta Car at 2.3 kpc, hypothetical SN at 2 kpc)
ax.axvspan(1.5e-3, 3.5e-3, color='gray', alpha=0.09, zorder=0)
# M31 distance band (785 kpc)
ax.axvspan(0.55, 1.15, color='gray', alpha=0.09, zorder=0)
# AGN / quasar distance range (1-10 Gpc)
ax.axvspan(1e3, 1e4, color='gray', alpha=0.09, zorder=0)

# ----- Lieu-Hillman threshold line -----------------------------
ax.axhline(sigma_LH, color='k', linestyle='--', lw=1.3, zorder=2)

# ----- foam model curves ---------------------------------------
ax.loglog(D_Mpc, sig_half, color=C_HALF,   lw=2.5, zorder=3,
          label=r'$\alpha = 1/2$  (random walk)')
ax.loglog(D_Mpc, sig_23,   color=C_TWOTHR, lw=2.5, zorder=3,
          label=r'$\alpha = 2/3$  (holographic)')

# ===========================================================
# Annotations
# ===========================================================
# imaging-excluded zone label: top-left corner, well above LH line
ax.text(2.2e-3, 1e-4, 'Imaging-excluded',
        ha='left', va='center', fontsize=10.5, color='#7a1d1d',
        style='italic', weight='bold', zorder=5)
ax.text(2.2e-3, 3e-5, '(Strehl / Lieu--Hillman, optical)',
        ha='left', va='top', fontsize=8.5, color='#7a1d1d',
        style='italic', zorder=5)

# Lieu-Hillman threshold: small label just above the dashed line, right side
ax.text(2.5e4, sigma_LH * 1.55,
        r'Lieu--Hillman fringe-erasure ($\sigma_\phi = 1$, $\lambda = 500\,$nm)',
        ha='right', va='bottom', fontsize=8.5, color='k', zorder=5)

# HBT band label: centered horizontally where no curve crosses (D ~ 30 Mpc)
ax.text(30, np.sqrt(sig_ground * sig_space_uv) * 1.65,
        'HBT Siegert-violation accessible',
        ha='center', va='center', fontsize=10.5, color=C_HBT_DK,
        weight='bold', zorder=5)
ax.text(30, np.sqrt(sig_ground * sig_space_uv) * 0.55,
        r'$\sigma_\phi \in [0.15,\,0.8]$:  ground at $\lambda=500\,$nm  to  space UV at $\lambda=250\,$nm',
        ha='center', va='center', fontsize=8.5, color=C_HBT_DK,
        style='italic', zorder=5)

# Galactic transient label
ax.text(np.sqrt(1.5e-3 * 3.5e-3), 4e-12, r'Galactic ($\eta$ Car / SN)',
        ha='center', va='center', fontsize=9, color=C_GRAY,
        rotation=90, zorder=5)
# M31 label
ax.text(np.sqrt(0.55 * 1.15), 4e-12, 'M31',
        ha='center', va='center', fontsize=9, color=C_GRAY,
        rotation=90, zorder=5)
# AGN / quasar band label: rotated, in the empty mid-vertical region
ax.text(np.sqrt(1e3 * 1e4), 4e-12, 'AGN / quasar',
        ha='center', va='center', fontsize=10, color=C_GRAY,
        rotation=90, zorder=5)

# alpha = 1 (conservative) caveat: small italic note at the bottom
ax.text(2.2e-3, 2.2e-18,
        r'$\alpha = 1$ (conservative): $\sigma_\ell = \ell_{\rm P} \approx 1.6\times10^{-35}$ m, far below scale',
        ha='left', va='bottom', fontsize=8.5, color=C_GRAY,
        style='italic', zorder=5)

# ===========================================================
# Axes
# ===========================================================
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r'Propagation distance $D$ [Mpc]', fontsize=11)
ax.set_ylabel(r'rms foam path-length amplitude $\sigma_\ell(D)$ [m]', fontsize=11)

# Legend: lower-right, where neither curve nor band is present
leg = ax.legend(loc='lower right', fontsize=10, framealpha=0.95,
                edgecolor='0.5', borderpad=0.6)
leg.set_zorder(6)

ax.grid(True, which='major', alpha=0.25)
ax.grid(True, which='minor', alpha=0.10)

plt.tight_layout()
plt.savefig('observability.pdf')
plt.savefig('observability.png', dpi=160)
print('Wrote observability.pdf and observability.png')
