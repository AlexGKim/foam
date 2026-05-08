r"""
Figure: SNR vs foam exponent alpha for 3C 273 and Mrk 766 (NLS1),
Kim 2025 baseline instrument (10 m, ε=0.39, σ_t=30 ps FWHM), T=10 hr.

Run:  python make_snr_alpha_figure.py
Outputs: snr_vs_alpha.pdf, snr_vs_alpha.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------
ell_P  = 1.616e-35    # Planck length [m]
Mpc    = 3.086e22     # 1 Mpc [m]
h_erg  = 6.626e-27    # erg·s
c_cgs  = 3.0e10       # cm/s
lam_cm = 6563e-8      # Hα [cm]
lam_m  = 6563e-10     # Hα [m]
E_phot = h_erg * c_cgs / lam_cm   # photon energy [erg]

# ---------------------------------------------------------------
# Baseline instrument (Kim et al. 2025)
# ---------------------------------------------------------------
d_tel     = 9.96          # telescope diameter [m]
A_tel_m2  = 77.9          # collecting area [m²]
A_tel_cm2 = A_tel_m2 * 1e4
eps       = 0.39          # end-to-end throughput
sig_t     = 13e-12        # timing resolution, rms [s]  (30 ps FWHM)
tau_c     = sig_t         # threshold filter: tau_c = sig_t (optimum)

# ---------------------------------------------------------------
# Sources
# ---------------------------------------------------------------
dlam = 1.1   # threshold filter width [Å]

# 3C 273 (Boroson & Green 1992): broad-line quasar
sources = [
    dict(label=r'3C\,273 (BLQ, $D=677$\,Mpc)',
         F=2e-12, FWHM=127.0, D_Mpc=677.0, color='#08306b'),
    # Mrk 766: bright NLS1, FWHM~1300 km/s = 28 Å at Hα (Kollatschny & Zetzl 2013)
    dict(label=r'Mrk\,766 (NLS1, $D=53$\,Mpc)',
         F=5e-13, FWHM=28.0,  D_Mpc=53.0,  color='#2ca02c'),
]

# ---------------------------------------------------------------
# Observation time
# ---------------------------------------------------------------
T_obs = 10 * 3600.0   # 10 hr [s]

# ---------------------------------------------------------------
# SNR vs alpha for each source
# ---------------------------------------------------------------
alpha = np.linspace(0.50, 0.75, 500)

for src in sources:
    D_m = src['D_Mpc'] * Mpc
    f_L = (2 / np.pi) * np.arctan(dlam / src['FWHM'])
    R   = src['F'] * f_L * A_tel_cm2 / E_phot * eps
    sigma_phi = (2 * np.pi / lam_m) * D_m**(1 - alpha) * ell_P**alpha
    S         = np.exp(-2.0) * (1.0 - np.exp(-2.0 * sigma_phi**2))
    src['SNR'] = S * R * np.sqrt(tau_c * T_obs)
    src['R']   = R

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

for src in sources:
    ax.semilogy(alpha, src['SNR'], color=src['color'], lw=2.2,
                label=src['label'])

ax.axhline(3, color='#d62728', lw=1.2, ls='--', label=r'$3\sigma$')
ax.axhline(1, color='#d62728', lw=0.8, ls=':',  label=r'$1\sigma$')

for a_mark, label in [(0.54, r'$\alpha=0.54$'), (2/3, r'$\alpha=2/3$')]:
    ax.axvline(a_mark, color='gray', lw=1.0, ls='--', alpha=0.6)
    ax.text(a_mark + 0.003, 2e-14, label,
            fontsize=9, color='gray', va='bottom')

ax.set_xlim(0.50, 0.75)
ax.set_ylim(1e-14, 1e3)
ax.set_xlabel(r'Foam exponent $\alpha$', fontsize=12)
ax.set_ylabel(r'SNR  ($T_{\rm obs} = 10\,{\rm hr}$)', fontsize=12)
ax.set_title(r'H$\alpha$, 10\,m baseline (Kim et al.\ 2025), $T=10$\,hr', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='major', alpha=0.25)
ax.grid(True, which='minor', alpha=0.08)

plt.tight_layout()
plt.savefig('snr_vs_alpha.pdf')
plt.savefig('snr_vs_alpha.png', dpi=160)
print('Wrote snr_vs_alpha.pdf and snr_vs_alpha.png')
