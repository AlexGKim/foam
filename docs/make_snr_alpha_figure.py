r"""
SNR vs foam exponent alpha: extended-source and point-source HBT variants.
Kim 2025 baseline instrument (10 m, ε=0.39, σ_t=30 ps FWHM), T=10 hr.

Run:  python make_snr_alpha_figure.py
Outputs:
  snr_vs_alpha_extended.pdf / .png   -- extended-source δg^(2)(0)
  snr_vs_alpha_pointsource.pdf / .png -- point-source S(τ_c)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
c_si   = 3.0e8        # m/s
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

# ---------------------------------------------------------------
# Sources
# ---------------------------------------------------------------
kpc_in_Mpc = 1e-3

# 3C 273 (Boroson & Green 1992): broad-line quasar
# Mrk 766 (Kollatschny & Zetzl 2013): bright NLS1, FWHM ~1300 km/s = 28 Å at Hα
# Eta Car: LBV giant eruption (m_V ~ -1), Hα FWHM ~700 km/s = 15 Å, D = 2.3 kpc
# Galactic SN IIn: m_V ~ -5 at peak, electron-scattering Lorentzian core
#   FWHM ~2500 km/s = 55 Å, D ~ 2 kpc.
# M31 SN IIn: same intrinsic luminosity as the galactic IIn, scaled to D = 785 kpc
#   (m_V ~ +7), F(Hα) ~ 5e-5 * (2/785)^2 ~ 3.2e-10 erg/s/cm^2.
SOURCES = [
    dict(label=r'3C\,273 (BLQ, 677\,Mpc)',
         F=2e-12,    FWHM=127.0, D_Mpc=677.0,              color='#08306b'),
    dict(label=r'Mrk\,766 (NLS1, 53\,Mpc)',
         F=5e-13,    FWHM=28.0,  D_Mpc=53.0,               color='#2ca02c'),
    dict(label=r'M31 SN\,IIn (785\,kpc)',
         F=3.2e-10,  FWHM=55.0,  D_Mpc=785.0*kpc_in_Mpc,  color='#9467bd'),
    dict(label=r'$\eta$\,Car eruption (2.3\,kpc)',
         F=1e-6,     FWHM=15.0,  D_Mpc=2.3*kpc_in_Mpc,    color='#ad3803'),
    dict(label=r'Galactic SN\,IIn (2\,kpc)',
         F=5e-5,     FWHM=55.0,  D_Mpc=2.0*kpc_in_Mpc,    color='#d62728'),
]

T_obs = 10 * 3600.0   # 10 hr [s]
alpha = np.linspace(0.50, 0.75, 500)


# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------

def photon_rate(src):
    """Photon rate through a 2×FWHM wide filter (captures ~70% of Lorentzian)."""
    f_L = (2 / np.pi) * np.arctan(2.0)   # (2/pi)*arctan(filter_half/fwhm_half) = (2/pi)*arctan(2)
    return src['F'] * f_L * A_tel_cm2 / E_phot * eps


def coherence_time(src):
    """Intrinsic Lorentzian line coherence time: tau = lambda^2 / (c * FWHM)."""
    return lam_m**2 / (c_si * src['FWHM'] * 1e-10)


def foam_params(src):
    """Return sigma_phi and sigma_ell arrays over alpha grid."""
    D_m = src['D_Mpc'] * Mpc
    sigma_phi = (2 * np.pi / lam_m) * D_m**(1 - alpha) * ell_P**alpha
    sigma_ell = sigma_phi * lam_m / (2 * np.pi)
    return sigma_phi, sigma_ell


def snr_from_signal(S_arr, R, tau_c):
    """
    SNR for signal array S_arr, photon rate R, and coherence time tau_c.
    Selects source-limited or detector-limited branch based on tau_c vs sig_t.
    Eqs. snr_src_ext/snr_src_pt (source-limited) and snr_det_ext/snr_det_pt
    (detector-limited) from paper §VII.A.
    """
    if tau_c > sig_t:
        return S_arr * R * np.sqrt(tau_c * T_obs)
    else:
        return (tau_c / sig_t) * S_arr * R * np.sqrt(sig_t * T_obs)


def make_axes():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0.50, 0.75)
    ax.set_ylim(1e-15, 1e8)
    ax.set_xlabel(r'Foam exponent $\alpha$', fontsize=12)
    ax.set_ylabel(r'SNR  ($T_{\rm obs} = 10\,{\rm hr}$)', fontsize=12)
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.08)
    for a_mark, lbl in [(0.54, r'$\alpha=0.54$'), (2/3, r'$\alpha=2/3$')]:
        ax.axvline(a_mark, color='gray', lw=1.0, ls='--', alpha=0.6)
        ax.text(a_mark + 0.003, 2e-15, lbl, fontsize=9, color='gray', va='bottom')
    return fig, ax


def finish_figure(ax, results, marker_label):
    for r in results:
        ax.semilogy(alpha, r['SNR'], color=r['color'], lw=2.2, label=r['label'])
    for r in results:
        ax.plot(r['alpha_lim'], r['SNR_lim'], 'o', color=r['color'],
                ms=7, mec='black', mew=0.8)
    validity_marker = mlines.Line2D([], [], marker='o', color='gray', ls='none',
                                    ms=7, mec='black', mew=0.8, label=marker_label)
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles + [validity_marker], labels_ + [marker_label],
              fontsize=9, loc='upper right',
              framealpha=0.95)
    plt.tight_layout()


# ---------------------------------------------------------------
# Figure 1: Extended-source variant
#   Observable: delta_g^(2)(0) = 1 - exp(2s^2)*erfc(sqrt(2)*s),  s = sigma_0/tau_L
#   Signal:     S = delta_g^(2)   (Eq. eq:deltag2)
#   SNR:        Eqs. snr_src_ext / snr_det_ext  (§VII.A, source/detector-limited)
#   Validity:   s = 0.3  (boundary of linear weak-foam approximation)
# ---------------------------------------------------------------

def compute_extended(sources):
    results = []
    for src in sources:
        tau_L     = coherence_time(src)
        R         = photon_rate(src)
        _, sigma_ell = foam_params(src)
        sigma_0   = np.sqrt(2) * sigma_ell / c_si   # Phi_eff = 0
        s         = sigma_0 / tau_L
        S         = 1.0 - np.exp(2*s**2) * erfc(np.sqrt(2)*s)
        snr       = snr_from_signal(S, R, tau_L)
        idx_lim   = np.argmin(np.abs(s - 0.3))
        results.append(dict(label=src['label'], color=src['color'],
                            SNR=snr, alpha_lim=alpha[idx_lim], SNR_lim=snr[idx_lim]))
    return results


fig, ax = make_axes()
ext_results = compute_extended(SOURCES)
finish_figure(ax, ext_results, r'$s=0.3$ (weak-foam limit)')
fig.savefig('snr_vs_alpha_extended.pdf')
fig.savefig('snr_vs_alpha_extended.png', dpi=160)
plt.close(fig)
print('Wrote snr_vs_alpha_extended.pdf and snr_vs_alpha_extended.png')


# ---------------------------------------------------------------
# Figure 2: Point-source variant
#   Observable: S(tau_c) = e^{-2} * (1 - exp(-2*sigma_phi^2))    (Eq. eq:Stc)
#   Foam param: sigma_phi = 2*pi*sigma_ell(D)/lambda              (Eq. eq:sigmaphi)
#   SNR:        Eqs. snr_src_pt / snr_det_pt  (§VII.A, source/detector-limited)
#               Structure identical to extended-source with tau_L -> tau_c.
#   Validity:   sigma_phi = 0.3  (boundary of quadratic weak-foam approximation
#               S(tau_c) ≈ 2*e^{-2}*sigma_phi^2; signal saturates at e^{-2} ≈ 0.135)
#   Filter:     same 2×FWHM wide filter as extended-source figure (for direct comparison)
# ---------------------------------------------------------------

def compute_pointsource(sources):
    results = []
    for src in sources:
        tau_c        = coherence_time(src)   # same as tau_L; wide filter used for fair comparison
        R            = photon_rate(src)
        sigma_phi, _ = foam_params(src)
        S            = np.exp(-2) * (1.0 - np.exp(-2 * sigma_phi**2))
        snr          = snr_from_signal(S, R, tau_c)
        idx_lim      = np.argmin(np.abs(sigma_phi - 0.3))
        results.append(dict(label=src['label'], color=src['color'],
                            SNR=snr, alpha_lim=alpha[idx_lim], SNR_lim=snr[idx_lim]))
    return results


fig, ax = make_axes()
pt_results = compute_pointsource(SOURCES)
finish_figure(ax, pt_results, r'$\sigma_\phi=0.3$ (weak-foam limit)')
fig.savefig('snr_vs_alpha_pointsource.pdf')
fig.savefig('snr_vs_alpha_pointsource.png', dpi=160)
plt.close(fig)
print('Wrote snr_vs_alpha_pointsource.pdf and snr_vs_alpha_pointsource.png')
