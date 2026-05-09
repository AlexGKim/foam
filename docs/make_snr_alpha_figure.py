r"""
Figure: SNR vs foam exponent alpha for 3C 273, Mrk 766 (NLS1), an
eta Car-class LBV giant eruption, a Galactic Type IIn supernova, and
an M31 Type IIn supernova, Kim 2025 baseline instrument
(10 m, ε=0.39, σ_t=30 ps FWHM), T=10 hr.

Run:  python make_snr_alpha_figure.py
Outputs: snr_vs_alpha.pdf, snr_vs_alpha.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

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

# kpc in Mpc
kpc_in_Mpc = 1e-3

# 3C 273 (Boroson & Green 1992): broad-line quasar
# Mrk 766 (Kollatschny & Zetzl 2013): bright NLS1, FWHM ~1300 km/s = 28 Å at Hα
# Eta Car: LBV giant eruption (m_V ~ -1), Hα FWHM ~700 km/s = 15 Å, D = 2.3 kpc
# Galactic SN IIn: m_V ~ -5 at peak, electron-scattering Lorentzian core
#   FWHM ~2500 km/s = 55 Å, D ~ 2 kpc.  CSM-interaction Hα emission throughout
#   the ~200 d bright phase makes it source-limited (cf. paper §IX.D).
# M31 SN IIn: same intrinsic luminosity as the galactic IIn, scaled to D = 785 kpc
#   (m_V ~ +7), F(Hα) ~ 5e-5 * (2/785)^2 ~ 3.2e-10 erg/s/cm^2.
sources = [
    dict(label=r'3C\,273 (BLQ, 677\,Mpc)',
         F=2e-12, FWHM=127.0, D_Mpc=677.0, color='#08306b'),
    dict(label=r'Mrk\,766 (NLS1, 53\,Mpc)',
         F=5e-13, FWHM=28.0,  D_Mpc=53.0,  color='#2ca02c'),
    dict(label=r'M31 SN\,IIn (785\,kpc)',
         F=3.2e-10, FWHM=55.0, D_Mpc=785.0*kpc_in_Mpc, color='#9467bd'),
    dict(label=r'$\eta$\,Car eruption (2.3\,kpc)',
         F=1e-6,  FWHM=15.0,  D_Mpc=2.3*kpc_in_Mpc, color='#ad3803'),
    dict(label=r'Galactic SN\,IIn (2\,kpc)',
         F=5e-5,  FWHM=55.0,  D_Mpc=2.0*kpc_in_Mpc, color='#d62728'),
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
    sigma_ell = sigma_phi * lam_m / (2 * np.pi)          # σ_ℓ(D) [m]
    c_si      = 3.0e8                                     # m/s
    sigma_0   = np.sqrt(2) * sigma_ell / c_si             # Φ_eff=0 limit [s]
    s         = sigma_0 / tau_c
    S         = 1.0 - np.exp(2*s**2) * erfc(np.sqrt(2)*s)
    sigma_g2  = 1.0 / (R * np.sqrt(tau_c * T_obs))   # Eq. hbt_sigma
    src['SNR'] = S / sigma_g2                          # Eq. hbt_snr
    src['R']   = R

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

for src in sources:
    ax.semilogy(alpha, src['SNR'], color=src['color'], lw=2.2,
                label=src['label'])


for a_mark, label in [(0.54, r'$\alpha=0.54$'), (2/3, r'$\alpha=2/3$')]:
    ax.axvline(a_mark, color='gray', lw=1.0, ls='--', alpha=0.6)
    ax.text(a_mark + 0.003, 2e-15, label,
            fontsize=9, color='gray', va='bottom')

ax.set_xlim(0.50, 0.75)
ax.set_ylim(1e-15, 1e8)
ax.set_xlabel(r'Foam exponent $\alpha$', fontsize=12)
ax.set_ylabel(r'SNR  ($T_{\rm obs} = 10\,{\rm hr}$)', fontsize=12)
ax.set_title(r'H$\alpha$, 10\,m baseline (Kim et al.\ 2025), $T=10$\,hr', fontsize=11)
ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.413, 0.98),
          framealpha=0.95)
ax.grid(True, which='major', alpha=0.25)
ax.grid(True, which='minor', alpha=0.08)

plt.tight_layout()
plt.savefig('snr_vs_alpha.pdf')
plt.savefig('snr_vs_alpha.png', dpi=160)
print('Wrote snr_vs_alpha.pdf and snr_vs_alpha.png')
