r"""
SNR vs foam exponent alpha: extended-source and point-source HBT variants.
Kim 2025 baseline instrument (10 m, ε=0.39, σ_t=30 ps FWHM), T=10 hr.

Run:  python make_snr_alpha_figure.py
Outputs:
  snr_vs_alpha.pdf / .png             -- extended-source δg^(2)(0)
  snr_vs_alpha_pointsource.pdf / .png -- point-source S(τ_L)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.special import erfcx
from scipy.integrate import quad

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
# Each source declares its own rest-frame line wavelength via src['lam_rest_A']
# (defaults to 6563 Å, Hα). For cosmological sources, foam_params() uses the
# OBSERVED-frame wavelength (λ_obs = λ_rest·(1+z)) and integrates the foam
# variance along the line of sight with a (1+z)² weighting in proper distance
# (equivalently, (1+z) weighting in comoving distance) to account for the
# z-dependent local wavelength at each foam cell.
LAM_REST_HA_M = 6563e-10      # default Hα rest-frame [m]

def _lam_rest_m(src):
    return src.get('lam_rest_A', 6563) * 1e-10

# ---------------------------------------------------------------
# Cosmology helpers (Planck 2018 flat ΛCDM)
# ---------------------------------------------------------------
H0_kmsMpc = 67.36
Om0       = 0.315
OL0       = 0.685
c_kmps    = 299792.458
D_H_Mpc   = c_kmps / H0_kmsMpc   # Hubble distance, ~4451 Mpc

def E_z(z):
    return np.sqrt(Om0*(1+z)**3 + OL0)

def D_C_Mpc(z):
    """Comoving distance [Mpc]"""
    return D_H_Mpc * quad(lambda zp: 1/E_z(zp), 0, z)[0]

def s_proper_Mpc(z):
    """Cumulative local proper distance traveled by the photon from observer
    to redshift z, in Mpc. s(z) = ∫₀^z D_H dz'/((1+z') E(z'))."""
    return D_H_Mpc * quad(lambda zp: 1/((1+zp)*E_z(zp)), 0, z)[0]

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

# All F(Hα), FWHM, D values match Table II in paper.tex (tab:source_agn).
# AGN luminosity distances use Planck 2018 cosmology (Aghanim et al. 2020,
# A&A 641, A6): H_0=67.36, Ω_m=0.315, Ω_Λ=0.685.
# Mrk 766 (Rodríguez-Ardila & Mazzalay 2006): bright NLS1, FWHM ~1300 km/s
#   = 28 Å at Hα [Kollatschny & Zetzl 2013], z = 0.013, D_L = 59 Mpc.
#   F(Hα) = 2.8e-12 erg/s/cm² is the nuclear total Hα flux tabulated by
#   R-A & M 2006 (281.1 ± 16.9 × 10⁻¹⁴), based on González Delgado &
#   Pérez 1996 spectrophotometry. Pop A classification (FWHM < 4000 km/s)
#   → Lorentzian-cored Hα profile (Sulentic 2002, Zamfir 2010) → the
#   linear-in-s scaling of δg^(2) at zero lag rigorously applies.
# PHL 1811 (Leighly et al. 2007, arXiv:0705.0940): NLS1 quasar, z = 0.192,
#   D_L = 971 Mpc (Planck 2018). FWHM(Hα) = 1752 km/s = 46 Å observed-frame.
#   F(Hα) = 100.6 × 10⁻¹⁴ = 1.006 × 10⁻¹² erg/s/cm² (Leighly 2007, Table 2).
#   The Hα profile is explicitly fit with a Lorentzian (Leighly 2007 §3),
#   providing direct empirical support for the linear-in-s scaling
#   requirement. Acts as a cosmological-distance Pop A complement to Mrk 766.
# Eta Car (Great Eruption peak, 1843): m_V ≈ -1 [Smith 2018,
#   2018MNRAS.480.1466S], D = 2.3 kpc, Hα FWHM ~700 km/s = 15 Å.
#   F(Hα) ≈ 5e-7 erg/s/cm² derives from m_V = -1 (Vega f_λ ≈ 9.1e-9
#   erg/s/cm²/Å) and an assumed EW(Hα) ~ 50 Å typical of LBVs in eruption;
#   uncertain by a factor of a few due to Homunculus scattering geometry.
# Type IIn SN (Galactic and M31): peak L(Hα) ≈ 1e41 erg/s as a
#   representative value for the IIn class [Taddia et al. 2013,
#   2013A&A...555A..10T] — consistent with EW(Hα) ~ 100 Å against the
#   r-band continuum implied by the sample-mean M_r = -19.16
#   [Ransome & Villar 2025, 2025ApJ...987...13R]. Scaled by inverse-square
#   distance: F(Hα, 2 kpc)   ≈ 2e-4 erg/s/cm²,
#                F(Hα, 778 kpc) ≈ 1.4e-9 erg/s/cm².
SOURCES = [
    dict(label=r'Mrk\,766 (NLS1, 59\,Mpc)',
         F=2.8e-12,  FWHM=28.0,  D_Mpc=58.6,               z=0.013,
         color='#08306b'),
    dict(label=r'PHL\,1811 (NLS1, 971\,Mpc)',
         F=1.006e-12, FWHM=45.7, D_Mpc=971.0,              z=0.192,
         color='#2ca02c'),
    dict(label=r'$\eta$\,Car eruption (2.3\,kpc)',
         F=5e-7,     FWHM=15.0,  D_Mpc=2.3*kpc_in_Mpc,     z=0.0,
         color='#ad3803'),
    dict(label=r'Galactic SN\,IIn (2\,kpc)',
         F=2e-4,     FWHM=55.0,  D_Mpc=2.0*kpc_in_Mpc,     z=0.0,
         color='#d62728'),
    dict(label=r'M31 SN\,IIn (778\,kpc)',
         F=1.4e-9,   FWHM=55.0,  D_Mpc=778.0*kpc_in_Mpc,   z=0.0,
         color='#9467bd'),
    # J1330-0905 image A (Persephone's Torch): brightest of four lensed images of
    # a quadruply-lensed quasar at z=2.2245 [Davies et al. 2026, arXiv:2604.13152].
    # Total system magnification ≈ 56; image A magnification ≈ 16.4; observed flux
    # ratios F_A:F_B:F_C:F_D = 1:0.74:0.70:0.08 give F_A ≈ 0.40 × system in the
    # blended GAIA blob. F = Lyα line flux on image A = 5.1×10⁻¹³ erg/s/cm²,
    # derived from the integrated GAIA DR3 XP spectrum (3700-4100 Å, source_id
    # 3629934529823678720) scaled by the image-A fraction.
    # The intrinsic broad-line FWHM in our adopted model is the *turbulent*
    # (Lorentzian) FWHM of 3800 km/s = 50 Å observed at λ_obs(Lyα) = 3920 Å,
    # which Kollatschny & Zetzl 2013 (A&A 549, A100) measured for the Lyα +
    # NV 1240 emitting region in reverberation-mapped AGN. The OBSERVED Lyα
    # profile is a Voigt convolution of this Lorentzian turbulent core with a
    # rotational Gaussian — the same model that gives Mrk 766's Hα the
    # Lorentzian core that the zero-lag observable requires. We adopt 52 Å
    # observed (4000 km/s, conservative) as the Lorentzian τ_L = 0.031 ps.
    # The Lorentzian-cored turbulent core means the cusp condition for the
    # zero-lag observable IS satisfied (with the Voigt reduction factor τ̃/σ_0,
    # close to unity in the weak-foam regime).
    # Lyα forest absorption preferentially attenuates the blue wing; the cusp
    # at τ=0 is set by the smooth Lorentzian core and survives moderate forest
    # absorption. The XP-measured flux already includes whatever forest
    # absorption is present.
    dict(label=r'J1330$-$0905 image\,A (lensed QSO, $z{=}2.22$)',
         F=5.1e-13,  FWHM=52.0,  D_Mpc=18149.0,            z=2.2245,
         lam_rest_A=1216.0,
         color='#e377c2'),
]

T_obs = 10 * 3600.0   # 10 hr [s]
alpha = np.linspace(0.50, 0.80, 500)


# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------

def photon_rate(src):
    """Photon rate through a 2×FWHM wide filter (captures ~70% of Lorentzian).
    Uses observed-frame photon energy hc/λ_obs since the detector receives
    redshifted photons."""
    lam_rest_cm = _lam_rest_m(src) * 100   # m to cm
    lam_obs_cm = lam_rest_cm * (1.0 + src['z'])
    E_phot_obs = h_erg * c_cgs / lam_obs_cm
    f_L = (2 / np.pi) * np.arctan(2.0)   # (2/π)·arctan(filter_half/fwhm_half) = (2/π)·arctan(2)
    return src['F'] * f_L * A_tel_cm2 / E_phot_obs * eps


def coherence_time(src):
    """Lorentzian line coherence time τ_L = λ_obs² / (π·c·Δλ_FWHM).

    Standard Wiener–Khinchin result for a Lorentzian intensity profile with
    g^(1)(τ) = exp(-|τ|/τ_L) (Mandel & Wolf §4; Goodman, Statistical Optics
    §6.5). Uses observed-frame λ because the detector receives redshifted
    light, and the observed FWHM in src['FWHM'] is likewise observed-frame.
    """
    lam_obs = _lam_rest_m(src) * (1.0 + src['z'])
    return lam_obs**2 / (np.pi * c_si * src['FWHM'] * 1e-10)


def foam_params(src):
    """Return sigma_phi and sigma_ell arrays over alpha grid.

    Implements the cosmological (1+z) correction to the foam phase variance.
    A foam cell at local redshift z imparts an observer-frame time delay
    δt_obs = (1+z)·δℓ_local/c, so the observer-frame phase fluctuation is
    δφ = (2π/λ_obs)·(1+z)·δℓ_local. For independent cell contributions:

        σ_φ² = (2π/λ_obs)² · ∫₀^L_local (1+z(s))² · dσ_ℓ²/ds · ds
             = (2π/λ_obs)² · A_α · 2(1-α) · ℓ_P^(2α) · ∫₀^D_C (1+z) · s(r_C)^(1-2α) · dr_C

    where s(r_C) = ∫₀^r_C dr_C'/(1+z(r_C')) is the cumulative LOCAL proper
    distance traveled by the photon up to comoving distance r_C, and the
    α-model differential is dσ_ℓ²/ds = A_α · 2(1-α) · s^(1-2α) · ℓ_P^(2α).

    The corresponding observer-frame path-length-equivalent variance is
        σ_ell² ≡ σ_φ² · λ_obs² / (2π)²,
    which is the input the σ_0 (extended-source) formula expects. (σ_0 enters
    the cusp signal as s = σ_0/τ_L through the observed-frame time delay,
    so σ_0 inherits the same cosmological correction as σ_φ.)

    For z ≈ 0 (Galactic sources) the formula reduces analytically to the
    flat-space form σ_φ² = (2π/λ_obs)² · A_α · D^(2(1-α)) · ℓ_P^(2α).
    A_α = 1 is adopted throughout.
    """
    z          = src.get('z', 0.0)
    lam_rest_m = _lam_rest_m(src)
    lam_obs_m  = lam_rest_m * (1.0 + z)
    if z < 1e-6:
        # Flat-space (Galactic) case: no cosmological correction needed.
        D_m = src['D_Mpc'] * Mpc
        sigma_phi = (2 * np.pi / lam_obs_m) * D_m**(1 - alpha) * ell_P**alpha
    else:
        # Cosmological case: integrate (1+z) · s(r_C)^(1-2α) over comoving distance
        # using redshift as the integration variable (dr_C = D_H·Mpc/E(z) dz).
        # Compute s(z) on a grid, then for each α perform the path integral.
        z_grid = np.linspace(0.0, z, 401)
        # Precompute s(z) on the grid (Mpc): cumulative trapezoidal
        ds_dz = D_H_Mpc / ((1 + z_grid) * E_z(z_grid))
        s_grid_Mpc = np.concatenate(([0.0], np.cumsum(0.5*(ds_dz[1:]+ds_dz[:-1])*np.diff(z_grid))))
        s_grid_m = s_grid_Mpc * Mpc        # meters
        # (1+z)/E(z) for the dr_C/dz part times (1+z) weighting collapsed:
        # integrand_dz = (1+z) · s^(1-2α) · D_H·Mpc/E(z)
        weight_dz = (1+z_grid) * D_H_Mpc * Mpc / E_z(z_grid)  # = (1+z) · dr_C/dz
        # For each α we need s_grid_m**(1 - 2α). At z=0, s=0; for α<1/2 the
        # integrand vanishes (s^positive), for α>1/2 it diverges as s^negative.
        # For the α∈[1/2, 1) range plotted here, use s_grid_m raised to (1-2α)<=0:
        # avoid 0^negative by skipping the z=0 endpoint (set its weight to 0,
        # which is correct since the integral is well-defined for α∈[1/2, 1)).
        alpha_arr = np.asarray(alpha)
        sigma_phi = np.zeros_like(alpha_arr, dtype=float)
        for i, a in enumerate(np.atleast_1d(alpha_arr)):
            exp_s = 1 - 2*a
            with np.errstate(divide='ignore', invalid='ignore'):
                if exp_s == 0:
                    integrand = weight_dz                              # α = 1/2 special case
                else:
                    integrand = weight_dz * s_grid_m**exp_s
                    integrand[0] = 0.0    # ignore z=0 endpoint singularity (integrable)
            integral_val = np.trapezoid(integrand, z_grid)
            sigma_phi_sq = (2*np.pi/lam_obs_m)**2 * 2*(1-a) * ell_P**(2*a) * integral_val
            sigma_phi[i] = np.sqrt(max(sigma_phi_sq, 0.0))
        sigma_phi = sigma_phi.reshape(alpha_arr.shape)
    # Observer-frame path-length-equivalent σ_ell: σ_φ · λ_obs / (2π).
    # σ_0 = √2 σ_ell/c uses this corrected σ_ell.
    sigma_ell = sigma_phi * lam_obs_m / (2 * np.pi)
    return sigma_phi, sigma_ell


def snr_from_signal(S_arr, R, tau_c):
    """
    SNR for signal array S_arr, photon rate R, and coherence time tau_c.
    Includes 1/4 prefactor for 50/50 beamsplitter with unpolarized light
    (Eq. snr_hbt). Selects source-limited or detector-limited branch.
    """
    if tau_c > sig_t:
        return 0.25 * S_arr * R * np.sqrt(tau_c * T_obs)
    else:
        return 0.25 * (tau_c / sig_t) * S_arr * R * np.sqrt(sig_t * T_obs)


def make_axes():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0.50, 0.80)
    ax.set_ylim(1e-15, 1e8)
    ax.set_xlabel(r'Foam exponent $\alpha$', fontsize=12)
    ax.set_ylabel(r'SNR  ($T_{\rm obs} = 10\,{\rm hr}$)', fontsize=12)
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.08)
    for a_mark, lbl in [(0.54, r'$\alpha=0.54$'), (2/3, r'$\alpha=2/3$')]:
        ax.axvline(a_mark, color='gray', lw=1.0, ls='--', alpha=0.6)
        ax.text(a_mark + 0.003, 2e-15, lbl, fontsize=9, color='gray', va='bottom')
    return fig, ax


def finish_figure(ax, results, marker_label, signal_label, ylim2=(1e-30, 1.0)):
    for r in results:
        ax.semilogy(alpha, r['SNR'], color=r['color'], lw=2.2, label=r['label'])
    for r in results:
        ax.plot(r['alpha_lim'], r['SNR_lim'], 'o', color=r['color'],
                ms=7, mec='black', mew=0.8)
    validity_marker = mlines.Line2D([], [], marker='o', color='gray', ls='none',
                                    ms=7, mec='black', mew=0.8, label=marker_label)
    solid_line = mlines.Line2D([], [], color='gray', lw=2.2, ls='-',
                               label=r'solid: SNR (left axis)')
    dashed_line = mlines.Line2D([], [], color='gray', lw=1.4, ls='--',
                                label=r'dashed: Siegert discrepancy (right axis)')
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles + [validity_marker, solid_line, dashed_line],
              labels_ + [marker_label, r'solid: SNR (left axis)',
                         r'dashed: Siegert discrepancy (right axis)'],
              fontsize=9, loc='upper right',
              framealpha=0.95)
    ax2 = ax.twinx()
    for r in results:
        ax2.semilogy(alpha, r['S'], color=r['color'], lw=1.4, ls='--')
    ax2.set_ylabel(signal_label, fontsize=12)
    ax2.set_ylim(*ylim2)
    plt.tight_layout()


# ---------------------------------------------------------------
# Figure 1: Extended-source variant
#   Observable: delta_g^(2)(0) = 1 - exp(2s^2)*erfc(sqrt(2)*s),  s = sigma_0/tau_L
#   Signal:     S = delta_g^(2)   (Eq. eq:deltag2)
#   SNR:        Eqs. snr_src_ext / snr_det_ext  (§VII.A, source/detector-limited)
#   Validity:   s = 0.3  (boundary of linear weak-foam approximation)
# ---------------------------------------------------------------

def compute_extended(sources):
    """Extended-source δg^(2)(0) and detector-limited SNR.

    σ_0² = 2 σ_ℓ²(D) (1 − Φ_eff) / c² (Eq. sigma0). All sources are evaluated
    in the fully resolved limit Φ_eff = 0 (maximally optimistic): the actual
    signal for sources whose angular size is smaller than the foam transverse
    coherence length r_⊥ is suppressed by √(1 − Φ_eff). For compact AGN BLRs
    this assumption is aggressive; it is appropriate for the SN/LBV targets
    whose physical extent at the source plane is large compared to plausible
    r_⊥ values.
    """
    results = []
    for src in sources:
        if 'extended' not in src.get('include', ('extended', 'pointsource')):
            continue
        tau_L     = coherence_time(src)
        R         = photon_rate(src)
        _, sigma_ell = foam_params(src)
        sigma_0   = np.sqrt(2) * sigma_ell / c_si   # Φ_eff = 0
        s         = sigma_0 / tau_L
        # Use weak-foam linear approx where erfcx rounds to 1 (s < ~1e-8)
        x = np.sqrt(2) * s
        S = np.where(x < 1e-8, 2.0 * np.sqrt(2.0 / np.pi) * s, 1.0 - erfcx(x))
        snr       = snr_from_signal(S, R, tau_L)
        idx_lim   = np.argmin(np.abs(s - 0.3))
        results.append(dict(label=src['label'], color=src['color'],
                            SNR=snr, S=S, alpha_lim=alpha[idx_lim], SNR_lim=snr[idx_lim]))
    return results


fig, ax = make_axes()
ext_results = compute_extended(SOURCES)
finish_figure(ax, ext_results,
              r'$s=0.3$ (weak-foam limit)',
              r'$|\mathcal{S}(0)|$',
              ylim2=(1e-16, 1.0))
fig.savefig('snr_vs_alpha.pdf')
fig.savefig('snr_vs_alpha.png', dpi=160)
plt.close(fig)
print('Wrote snr_vs_alpha.pdf and snr_vs_alpha.png')


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
    """Point-source Siegert discrepancy S(τ_L) and detector-limited SNR.

    Evaluated in the dynamical-foam limit C_α(τ_L; D) → 0 of Eq. Stc, so
    S(τ_L) = e^(-2) (1 - exp(-2 σ_φ²)). For quasi-static foam the signal
    is suppressed by an additional factor [1 - C_α(τ_L)].
    """
    results = []
    for src in sources:
        if 'pointsource' not in src.get('include', ('extended', 'pointsource')):
            continue
        tau_c        = coherence_time(src)   # same as tau_L; wide filter used for fair comparison
        R            = photon_rate(src)
        sigma_phi, _ = foam_params(src)
        # Use weak-foam quadratic approx where exp rounds to 1 (sigma_phi < ~1e-8)
        S            = np.exp(-2) * (-np.expm1(-2 * sigma_phi**2))
        snr          = snr_from_signal(S, R, tau_c)
        idx_lim      = np.argmin(np.abs(sigma_phi - 0.3))
        results.append(dict(label=src['label'], color=src['color'],
                            SNR=snr, S=S, alpha_lim=alpha[idx_lim], SNR_lim=snr[idx_lim]))
    return results


fig, ax = make_axes()
pt_results = compute_pointsource(SOURCES)
finish_figure(ax, pt_results,
              r'$\sigma_\phi=0.3$ (weak-foam limit)',
              r'$|\mathcal{S}(\tau_L)|$',
              ylim2=(1e-26, 1e-3))
fig.savefig('snr_vs_alpha_pointsource.pdf')
fig.savefig('snr_vs_alpha_pointsource.png', dpi=160)
plt.close(fig)
print('Wrote snr_vs_alpha_pointsource.pdf and snr_vs_alpha_pointsource.png')
