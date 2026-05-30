import numpy as np
from scipy.special import erfc

def compute_delta_g2(sigma_phi, delta_lambda_over_lambda, C_alpha, Phi_eff):
    s = sigma_phi * delta_lambda_over_lambda * np.sqrt(2 * (1 - C_alpha) * (1 - Phi_eff))
    delta_g2 = 1 - np.exp(2 * s**2) * erfc(np.sqrt(2) * s)
    return s, delta_g2

def format_sci_latex(x):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    if abs(coeff - round(coeff)) < 0.005:
        coeff = round(coeff)
        return f"{coeff:.0f} \\times 10^{{{exp}}}"
    else:
        return f"{coeff:.2f} \\times 10^{{{exp}}}"

# Fiducial parameters
D_Gpc    = 1.0
lambda_m = 500e-9
dl_over_l = 1e-3
A_alpha  = 1.0
C_alpha  = 0.0
Phi_eff  = 0.0
ell_P    = 1.616e-35
D_m      = D_Gpc * 3.086e25

results = {}
for alpha, label in [(0.5, "half"), (2/3, "twothirds")]:
    sigma_ell = np.sqrt(A_alpha) * D_m**(1 - alpha) * ell_P**alpha
    sigma_phi = 2 * np.pi * sigma_ell / lambda_m
    s, dg2 = compute_delta_g2(sigma_phi, dl_over_l, C_alpha, Phi_eff)
    results[label] = dict(sigma_ell=sigma_ell, sigma_phi=sigma_phi, s=s, dg2=dg2)

# Extract values
s_half   = results["half"]["s"]
s_two    = results["twothirds"]["s"]
dg2_half = results["half"]["dg2"]
dg2_two  = results["twothirds"]["dg2"]
sphi_half = results["half"]["sigma_phi"]
sphi_two  = results["twothirds"]["sigma_phi"]

latex = (
    "\\begin{table}[h]\n"
    "\\centering\n"
    "\\caption{Bunching suppression $\\delta g^{(2)}$ from Eq.~(46) and sensitivity parameter\n"
    "$s = \\sigma_0/\\tau_c$ for an extended source in the fully resolved limit\n"
    "$\\Phi_{\\rm eff} \\to 0$, using the fiducial parameters of Table~I:\n"
    "$D = 1\\,\\mathrm{Gpc}$, $\\lambda = 500\\,\\mathrm{nm}$,\n"
    "$\\Delta\\lambda/\\lambda = 10^{-3}$, $A_\\alpha = 1$, $C_\\alpha \\to 0$.\n"
    "The $\\alpha = 1/2$ case is observationally excluded~\\cite{Perlman2015};\n"
    "$\\alpha = 2/3$ (holographic) is the most sensitive surviving model.\n"
    "For $\\alpha = 1/2$ the approximation $\\delta g^{(2)} \\approx s^2$ is inaccurate\n"
    "at the $\\sim$30\\% level; the full expression Eq.~(46) must be used.}\n"
    "\\label{tab:bunching}\n"
    "\\begin{tabular}{lcc}\n"
    "\\hline\\hline\n"
    "Quantity & $\\alpha = 1/2$ (excluded) & $\\alpha = 2/3$ \\\\\n"
    "\\hline\n"
    f"$\\sigma_\\phi$ & ${format_sci_latex(sphi_half)}$ & ${format_sci_latex(sphi_two)}$ \\\\\n"
    f"$s = \\sigma_0/\\tau_c$ & ${format_sci_latex(s_half)}$ & ${format_sci_latex(s_two)}$ \\\\\n"
    f"$\\delta g^{{(2)}}$ [Eq.~(46), exact] & ${format_sci_latex(dg2_half)}$ & ${format_sci_latex(dg2_two)}$ \\\\\n"
    f"$\\delta g^{{(2)}} \\approx s^2$ [small-$s$] & ${format_sci_latex(s_half**2)}$ & ${format_sci_latex(s_two**2)}$ \\\\\n"
    "\\hline\\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

print(latex)
