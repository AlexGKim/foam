r"""
Figure: HBT SNR per sqrt(T/hr) vs. propagation distance D for three astrophysical
source types plus the Sun, for the holographic foam exponent alpha = 2/3.

Run:  python make_snr_figure.py
Outputs: snr_vs_D.pdf, snr_vs_D.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Physical constants and instrument
# ---------------------------------------------------------------
ell_P   = 1.616e-35       # Planck length [m]
Mpc     = 3.086e22        # 1 Mpc [m]
AU      = 1.496e11        # 1 AU [m]
pc      = 3.086e16        # 1 pc [m]
lam     = 6.563e-7        # Hα wavelength [m]
tau_c   = 13e-12          # coherence time = σ_t (threshold filter) [s]
eps     = 0.39            # end-to-end throughput
d_tel   = 10.0            # telescope diameter [m]
A_tel   = np.pi * (d_tel/2)**2   # collecting area [m²]

sqrt_tc_hr = np.sqrt(tau_c * 3600.0)

# ---------------------------------------------------------------
# Distance grid [Mpc]:  1 AU to 10 Gpc
# ---------------------------------------------------------------
AU_in_Mpc = AU / Mpc                    # ≈ 4.85e-12 Mpc
D_Mpc = np.logspace(-6, 4, 1000)
D_m   = D_Mpc * Mpc

# ---------------------------------------------------------------
# Foam signal S(tau_c), alpha = 2/3 only
# ---------------------------------------------------------------
def sigma_phi(D_m, alpha):
    return (2 * np.pi / lam) * D_m**(1 - alpha) * ell_P**alpha

def S_tc(D_m, alpha):
    sp = sigma_phi(D_m, alpha)
    return np.exp(-2.0) * (-np.expm1(-2.0 * sp**2))

S_23 = S_tc(D_m, 2.0/3.0)

# ---------------------------------------------------------------
# Photon rates R(D)  [counts/s on 10 m, ε=0.39, Δλ=1.1 Å filter]
# ---------------------------------------------------------------

# Reference: V=0 (Vega) star at D0 = 7.7 pc
# Vega Hα flux ≈ 726 phot/cm²/s/Å through Δλ = 1.1 Å
D0_star_m = 7.7 * pc
R0_star   = 726.0 * 1.1 * (A_tel * 1e4) * eps   # cps at D0

# -- Sun (M_V = 4.83, D = 1 AU): same distance-scaling formula as other sources
D_sun_m   = 1.0 * AU
D_sun_Mpc = D_sun_m / Mpc
M_V_ref   = 0.0 - 5.0 * np.log10(7.7 / 10.0)   # abs mag of V=0 reference at 7.7 pc
M_V_sun   = 4.83
flux_ratio_sun = 10.0 ** ((M_V_ref - M_V_sun) / 2.5)
R_sun_scalar = R0_star * flux_ratio_sun * (D0_star_m / D_sun_m) ** 2

# -- Bright star (fiducial V=0, scaled by distance)
R_star = R0_star * (D0_star_m / D_m)**2

# -- Type Ia SN at peak (M_V = -19.3; 19.3 mag brighter than V=0)
flux_ratio_SNIa = 10**(19.3 / 2.5)
R_SNIa = R0_star * flux_ratio_SNIa * (D0_star_m / D_m)**2

# -- Broad-line AGN / quasar (3C 273-class, Hα Lorentzian)
#    R(677 Mpc) = 1110 cps from paper
R_AGN = 1110.0 * (677.0 / D_Mpc)**2

# ---------------------------------------------------------------
# SNR per sqrt(T/hr)
# ---------------------------------------------------------------
def snr_per_sqrthr(S, R):
    return S * R * sqrt_tc_hr

snr_star = snr_per_sqrthr(S_23, R_star)
snr_snia = snr_per_sqrthr(S_23, R_SNIa)
snr_agn  = snr_per_sqrthr(S_23, R_AGN)

# Sun point
S_sun = S_tc(np.array([D_sun_m]), 2.0/3.0)[0]
snr_sun = snr_per_sqrthr(S_sun, R_sun_scalar)

# ---------------------------------------------------------------
# Colors
# ---------------------------------------------------------------
C_STAR  = '#08306b'   # deep navy blue
C_SNIA  = '#d62728'   # red
C_AGN   = '#2ca02c'   # green
C_SUN   = '#e6a817'   # amber/gold

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))

snr_max = max(snr_star[0], snr_snia[0], snr_agn[0], snr_sun)
ymax = 10 ** (np.ceil(np.log10(snr_max)) + 0.5)
ymin = 1e-32
xmin = 1e-6
xmax = 1e4

# --- source distance ranges ------------------------------------
ax.axvspan(AU_in_Mpc, 3.0,   color=C_STAR, alpha=0.05, zorder=0)
ax.axvspan(1.0,        5e3,   color=C_SNIA, alpha=0.05, zorder=0)
ax.axvspan(10.0,       1e4,   color=C_AGN,  alpha=0.05, zorder=0)

# --- foam curves (alpha = 2/3 only) ----------------------------
lw = 2.2
ax.loglog(D_Mpc, snr_star, color=C_STAR, lw=lw, ls='-', zorder=3,
          label=r'Bright star ($V=0$)')
ax.loglog(D_Mpc, snr_snia, color=C_SNIA, lw=lw, ls='-', zorder=3,
          label=r'SNIa at peak ($M_V=-19.3$)')
ax.loglog(D_Mpc, snr_agn,  color=C_AGN,  lw=lw, ls='-', zorder=3,
          label=r'AGN/quasar (H$\alpha$, 3C\,273-class)')

# --- Sun horizontal line ---------------------------------------
ax.axhline(snr_sun, color=C_SUN, linestyle='--', lw=1.4, zorder=4,
           label=r'Sun ($V=-26.7$, $D=1\,$AU)')
ax.text(xmax * 0.85, snr_sun * 2.5, r'Sun ($D=1\,$AU)',
        ha='right', va='bottom', fontsize=8.5, color=C_SUN, zorder=5)

# --- source-range labels ---------------------------------------
ax.text(3e-4,  ymax * 0.35, 'Stars',  color=C_STAR, fontsize=9, ha='center',
        alpha=0.8, style='italic')
ax.text(7e1,   ymax * 0.35, 'SNIa',   color=C_SNIA, fontsize=9, ha='center',
        alpha=0.8, style='italic')
ax.text(3e3,   ymax * 0.35, 'AGN',    color=C_AGN,  fontsize=9, ha='center',
        alpha=0.8, style='italic')

# --- M31 distance line (770 kpc) -------------------------------
D_M31_Mpc = 0.770
ax.axvline(D_M31_Mpc, color='#888888', linestyle='--', lw=1.2, zorder=2)
ax.text(D_M31_Mpc * 1.12, ymin * 3e2, 'M31\n(770 kpc)',
        ha='left', va='bottom', fontsize=8.5, color='#555555', zorder=5)

# --- alpha label -----------------------------------------------
ax.text(0.97, 0.97, r'Holographic foam: $\alpha = 2/3$',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, style='italic', color='#444444',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.8, alpha=0.9),
        zorder=6)

# --- axes ------------------------------------------------------
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r'Propagation distance $D$ [Mpc]', fontsize=11)
ax.set_ylabel(r'SNR / $\sqrt{T\,/\,\mathrm{hr}}$', fontsize=11)


# --- legend ----------------------------------------------------
leg = ax.legend(loc='lower left', fontsize=9, framealpha=0.95,
                edgecolor='0.5', borderpad=0.6)
leg.set_zorder(8)

ax.grid(True, which='major', alpha=0.22)
ax.grid(True, which='minor', alpha=0.08)

plt.tight_layout()
plt.savefig('snr_vs_D.pdf')
plt.savefig('snr_vs_D.png', dpi=160)
print('Wrote snr_vs_D.pdf and snr_vs_D.png')
