"""
================================================================================
  Coherent Multidimensional Spectroscopy — Response Function Calculator
  Multimode Displaced Harmonic Oscillator Model
  Blocks 1–3 (Blocks 4–5: Herzberg-Teller, to be added)
================================================================================

USAGE:
    python response_functions.py input.txt

INPUT FILE FORMAT (see example_input.txt for a template):
    # Lines starting with # are comments
    M = 4
    kets = 0 1 2 1
    bras = 0 1 2 1
    # Parameters
    T = 300              # Temperature in Kelvin (default 300)
    tau_dep = 100000.0   # Dephasing time in femtoseconds
    tau_rel = 500000.0   # Relaxation time in femtoseconds
    epsilon = 0.0 100.0 200.0   # Electronic energies in meV (N values)
    # Dipole matrix elements mu[j][l] (real, symmetric): N_elec x N_elec matrix
    # One row per electronic state j, columns are states l
    mu = 0.0 1.0 0.5
         1.0 0.0 0.8
         0.5 0.8 0.0
    omega_vib = 50.0 120.0      # Vibrational energies in meV (Nm values)
    # Displacements: z[j][m] for electronic state j, mode m
    # One row per electronic state
    z = 0.0 0.0
        0.5 0.3
        0.8 0.6
    # Time grid
    t_max = 500.0        # fs, upper limit for time integration
    N_t   = 512          # number of time points per axis
    # Fixed times: specify which M-2 of the M time indices (1-based) are fixed
    # and at what values in fs. Leave empty if M=2.
    fixed_times = 2:100.0  3:0.0
================================================================================
"""

import sys
import os
import re
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Physical constants ─────────────────────────────────────────────────────────
HBAR_meV_fs = 0.6582119569  # ℏ in meV·fs  (so E[meV]*t[fs]/hbar gives radians)
KB_meV_K    = 0.08617333    # k_B in meV/K

# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 1 — Parse input, identify response function, draw Feynman diagram
# ══════════════════════════════════════════════════════════════════════════════

def parse_ht_input(params, content_clean, N_elec, Nm):
    """
    Parse HT-related input fields into params.
    Reads:
      use_HT1, use_HT2, use_HT3  (bool flags, default False)
      mu1_1 .. mu1_Nm             (N_elec x N_elec matrices, only if use_HT1 or use_HT2)
      mu2_1 .. mu2_Nm             (N_elec x N_elec matrices, only if use_HT3)
    """
    def get_val(key):
        m = re.search(rf"^\s*{key}\s*=\s*(.+?)(?:\n|$)", content_clean, re.MULTILINE)
        return m.group(1).strip() if m else None

    def parse_flag(key):
        val = get_val(key)
        if val is None:
            return False
        return val.strip().lower() not in ("0", "false", "no", "off")

    def parse_mu_list(prefix, required):
        """
        Parse Nm matrices named prefix_1 .. prefix_Nm.
        Each is an N_elec x N_elec block of floats.
        Returns list of Nm numpy arrays, or None if not required.
        """
        if not required:
            return None
        matrices = []
        for mi in range(1, Nm + 1):
            key = f"{prefix}_{mi}"
            # Search for multi-line block: key = <floats>
            pat = rf"{key}\s*=\s*([\d\s.\-+eE]+?)(?=\w+\s*=|$)"
            match = re.search(pat, content_clean, re.DOTALL)
            if match is None:
                raise ValueError(
                    f"{key} not found in input, but {prefix} is required "
                    f"(use_HT1/use_HT2/use_HT3 flag is set)."
                )
            vals = list(map(float, match.group(1).split()))
            if len(vals) != N_elec * N_elec:
                raise ValueError(
                    f"{key} must have N_elec^2={N_elec**2} values, got {len(vals)}."
                )
            matrices.append(np.array(vals).reshape(N_elec, N_elec))
        return matrices

    params["use_HT1"] = parse_flag("use_HT1")
    params["use_HT2"] = parse_flag("use_HT2")
    params["use_HT3"] = parse_flag("use_HT3")

    need_mu1 = params["use_HT1"] or params["use_HT2"]
    need_mu2 = params["use_HT3"]

    params["mu1"] = parse_mu_list("mu1", need_mu1)   # list of Nm matrices or None
    params["mu2"] = parse_mu_list("mu2", need_mu2)   # list of Nm matrices or None

    return params


def parse_input(filename):
    """Read the input file and return a parameter dictionary."""
    params = {}
    with open(filename) as f:
        content = f.read()

    # Strip comments
    lines = [re.sub(r"#.*", "", l).strip() for l in content.splitlines()]
    content_clean = "\n".join(lines)

    def get_val(key):
        """Return the raw string value for a key = value entry."""
        m = re.search(rf"^\s*{key}\s*=\s*(.+?)(?:\n|$)", content_clean, re.MULTILINE)
        return m.group(1).strip() if m else None

    # Order M
    val = get_val("M")
    if val is None:
        raise ValueError("M not found in input file.")
    params["M"] = int(val)
    M = params["M"]

    # kets and bras (M+1 values each: index 0..M)
    val = get_val("kets")
    if val is None:
        raise ValueError("kets not found.")
    params["kets"] = list(map(int, val.split()))
    val = get_val("bras")
    if val is None:
        raise ValueError("bras not found.")
    params["bras"] = list(map(int, val.split()))

    if len(params["kets"]) != M + 1:
        raise ValueError(f"kets must have M+1={M+1} entries.")
    if len(params["bras"]) != M + 1:
        raise ValueError(f"bras must have M+1={M+1} entries.")

    # Each time step i must carry an arrow on at most one side:
    # a ket change and a bra change cannot happen simultaneously.
    kets = params["kets"]
    bras = params["bras"]
    for i in range(1, M + 1):
        if (kets[i] != kets[i - 1]) and (bras[i] != bras[i - 1]):
            raise ValueError(
                f"At time step i={i} both ket and bra change state simultaneously. "
                f"Each arrow must act on exactly one side (ket or bra)."
            )

    # Temperature
    val = get_val("T")
    params["T"] = float(val) if val else 300.0

    # Dephasing and relaxation times (input in ps, stored in fs)
    val = get_val("tau_dep")
    params["tau_dep"] = float(val) if val else 100000.0   # fs
    val = get_val("tau_rel")
    params["tau_rel"] = float(val) if val else 100000.0   # fs

    # N_elec: derived from the highest state index in kets/bras, plus 1
    N_elec = max(params["kets"] + params["bras"]) + 1
    params["N_elec"] = N_elec

    # Electronic energies (must have exactly N_elec values)
    val = get_val("epsilon")
    if val is None:
        raise ValueError("epsilon not found.")
    params["epsilon"] = list(map(float, val.split()))
    if len(params["epsilon"]) != N_elec:
        raise ValueError(
            f"epsilon must have N_elec={N_elec} values "
            f"(= max state index + 1), got {len(params['epsilon'])}."
        )

    # Vibrational energies
    val = get_val("omega_vib")
    if val is None:
        raise ValueError("omega_vib not found.")
    params["omega_vib"] = list(map(float, val.split()))
    Nm = len(params["omega_vib"])

    # Displacements z[j][m]: parse multi-line block after "z ="
    m_z = re.search(r"z\s*=\s*([\d\s.\-+eE\n]+?)(?=\w+\s*=|$)", content_clean, re.DOTALL)
    if m_z is None:
        raise ValueError("z displacements not found.")
    z_vals = list(map(float, m_z.group(1).split()))
    if len(z_vals) != N_elec * Nm:
        raise ValueError(f"z must have N_elec*Nm = {N_elec}*{Nm} = {N_elec*Nm} values, got {len(z_vals)}.")
    params["z"] = np.array(z_vals).reshape(N_elec, Nm)

    # Dipole matrix mu_0[j][l]: N_elec x N_elec real symmetric matrix
    # Accept both 'mu_0' and legacy 'mu' key names
    m_mu = re.search(r"mu_0\s*=\s*([\d\s.\-+eE]+?)(?=\w+\s*=|$)", content_clean, re.DOTALL)
    if m_mu is None:
        m_mu = re.search(r"(?<!mu_[12])mu\s*=\s*([\d\s.\-+eE]+?)(?=\w+\s*=|$)", content_clean, re.DOTALL)
    if m_mu is None:
        raise ValueError("mu_0 (permanent dipole matrix) not found.")
    mu_vals = list(map(float, m_mu.group(1).split()))
    if len(mu_vals) != N_elec * N_elec:
        raise ValueError(f"mu_0 must have N_elec^2 = {N_elec}^2 = {N_elec**2} values, got {len(mu_vals)}.")
    params["mu_0"] = np.array(mu_vals).reshape(N_elec, N_elec)

    # Time grid
    val = get_val("t_max")
    params["t_max"] = float(val) if val else 500.0
    val = get_val("N_t")
    params["N_t"] = int(val) if val else 256

    # Fixed times: format  "i:value  j:value  ..."
    val = get_val("fixed_times")
    fixed = {}
    if val and val.strip():
        for tok in val.split():
            if ":" in tok:
                idx_s, tv_s = tok.split(":")
                fixed[int(idx_s)] = float(tv_s)
    params["fixed_times"] = fixed  # keys are 1-based time indices

    # Optional frequency plot range: "omega_min:omega_max" in meV
    # If omitted, the full FFT range is used.
    val = get_val("freq_range")
    if val and val.strip():
        parts = val.strip().split(":")
        params["freq_range"] = (float(parts[0]), float(parts[1]))
    else:
        params["freq_range"] = None

    # Herzberg-Teller flags and dipole operators
    parse_ht_input(params, content_clean, N_elec, Nm)

    # Whether to compile the .tex file to PDF (default: yes)
    val = get_val("compile_latex")
    params["compile_latex"] = (val.strip().lower() not in ("0","false","no","off"))\
        if val else True

    return params


def build_ek_tau(params):
    """
    Build e_k sequence and tau_k coefficients by walking the Feynman diagram
    clockwise: up the ket side (k=1..M), then down the bra side (k=M-1..1).

    Algorithm (translated from Fortran):
      Forward (ket, k=1..M-1):
        ket changes → start new tau_l = +t_k; e_l = kets[k]
        ket same    → accumulate tau_l += +t_k  (if l != 0)
      Ket during t_M (if kets[M] != 0):
        ket changes → start new tau_l = +t_M; e_l = kets[M]
        ket same    → accumulate tau_l += +t_M
      Middle (final emission, always):
        l += 1; tau_l = -t_M; e_l = bras[M]
      Backward (bra, k=M-1..1):
        bra changes → start new tau_l = -t_k; e_l = bras[k]
        bra same    → accumulate tau_l -= t_k

    Returns
    -------
    e_seq      : list of int, length K <= M
    tau_coeffs : list of dict, length K
                 tau_coeffs[k] = {t_index: coefficient} for tau_{k+1}
    """
    M    = params["M"]
    kets = params["kets"]
    bras = params["bras"]

    tau_coeffs = [{} for _ in range(M + 2)]
    e_seq      = [0]  * (M + 2)
    l = 0

    # Forward loop k=1..M-1
    for k in range(1, M):
        if kets[k] != kets[k-1]:
            l += 1
            if l > M: break
            tau_coeffs[l][k] = tau_coeffs[l].get(k, 0) + 1
            e_seq[l] = kets[k]
        elif l != 0:
            tau_coeffs[l][k] = tau_coeffs[l].get(k, 0) + 1

    # Ket state during t_M (only if nonzero)
    if kets[M] != 0:
        if kets[M] != kets[M-1]:
            l += 1
            if l <= M:
                tau_coeffs[l][M] = tau_coeffs[l].get(M, 0) + 1
                e_seq[l] = kets[M]
        elif l != 0:
            tau_coeffs[l][M] = tau_coeffs[l].get(M, 0) + 1

    # Middle: final emission (always)
    l += 1
    if l <= M:
        tau_coeffs[l][M] = tau_coeffs[l].get(M, 0) - 1
        e_seq[l] = bras[M]

    # Backward loop k=M-1..1
    for k in range(M-1, 0, -1):
        if l > M: break
        if bras[k] != bras[k+1]:
            l += 1
            if l > M: break
            tau_coeffs[l][k] = tau_coeffs[l].get(k, 0) - 1
            e_seq[l] = bras[k]
        else:
            if l <= M:
                tau_coeffs[l][k] = tau_coeffs[l].get(k, 0) - 1

    K = min(l, M)
    return e_seq[1:K+1], tau_coeffs[1:K+1]




def compute_x(params):
    """
    x = 0 if the number of arrows on the bra side (bra-state changes among
    bra_1..bra_M) is even, x = 1 if it is odd.
    """
    bras = params["bras"]
    M    = params["M"]
    n_arrows_bra = sum(1 for i in range(1, M + 1) if bras[i] != bras[i - 1])
    return n_arrows_bra % 2


def draw_feynman_diagram(params, e_seq, tau_coeffs, outfile="feynman_diagram.png"):
    """
    Draw a double-sided Feynman diagram for the specified ket/bra sequence.
    Ket side on the left, bra side on the right, time running upward.
    Interaction arrows are labeled with t_i.
    """
    M    = params["M"]
    kets = params["kets"]
    bras = params["bras"]

    fig, ax = plt.subplots(figsize=(5, 0.9 * (M + 3)))
    ax.set_xlim(-1.8, 3.8)
    ax.set_ylim(-0.6, M + 2.3)
    ax.axis("off")
    ax.set_title("Double-Sided Feynman Diagram", fontsize=13, pad=12)

    x_ket     = 0.0
    x_bra     = 2.0
    x_out_ket = x_ket - 1.0   # outside point for ket arrows (to the left)
    x_out_bra = x_bra + 1.0   # outside point for bra arrows (to the right)

    # ── Vertical time lines extending to y = M+1 ──────────────────────────
    ax.annotate("", xy=(x_ket, M + 1.7), xytext=(x_ket, -0.2),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.annotate("", xy=(x_bra, M + 1.7), xytext=(x_bra, -0.2),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.text(x_ket, -0.45, "ket", ha="center", fontsize=10)
    ax.text(x_bra, -0.45, "bra", ha="center", fontsize=10)
    ax.text(1.0, M + 2.0, "time ↑", ha="center", fontsize=9, color="gray")

    # ── Dashed horizontal lines at y = 0, 1, ..., M, M+1 ─────────────────
    # y = M+1 is the extra line that closes the last waiting interval t_M.
    for i in range(M + 2):
        ax.plot([x_ket, x_bra], [i, i], "k--", lw=0.7, alpha=0.4)

    # ── State labels midway between dashed lines, at y = i + 0.5 ────────
    # This places them in the same vertical band as the waiting-time labels
    # and well clear of the interaction arrows anchored at integer y values.
    for i in range(M + 1):
        ax.text(x_ket - 0.12, i + 0.5, f"|{kets[i]}⟩", ha="right", va="center",
                fontsize=10, color="#1a6faf")
        ax.text(x_bra + 0.12, i + 0.5, f"⟨{bras[i]}|", ha="left", va="center",
                fontsize=10, color="#a63c1e")

    # ── Interaction arrows i = 1..M ───────────────────────────────────────
    # Arrow i is anchored at y = i (the dashed line at that level).
    # Excitation   (state increases): tail at (x_out, i-0.5), tip at (x_line, i)  → inward  & upward
    # Deexcitation (state decreases): tail at (x_line, i),    tip at (x_out, i+0.5) → outward & upward
    for i in range(1, M + 1):
        y_line = i
        y_low  = i - 0.5
        y_high = i + 0.5

        # ── Ket side ──────────────────────────────────────────────────────
        if kets[i] != kets[i - 1]:
            if kets[i] > kets[i - 1]:
                ax.annotate("", xy=(x_ket, y_line), xytext=(x_out_ket, y_low),
                            arrowprops=dict(arrowstyle="-|>", color="#1a6faf", lw=1.5))
            else:
                ax.annotate("", xy=(x_out_ket, y_high), xytext=(x_ket, y_line),
                            arrowprops=dict(arrowstyle="-|>", color="#1a6faf", lw=1.5))

        # ── Bra side ──────────────────────────────────────────────────────
        if bras[i] != bras[i - 1]:
            if bras[i] > bras[i - 1]:
                ax.annotate("", xy=(x_bra, y_line), xytext=(x_out_bra, y_low),
                            arrowprops=dict(arrowstyle="-|>", color="#a63c1e", lw=1.5))
            else:
                ax.annotate("", xy=(x_out_bra, y_high), xytext=(x_bra, y_line),
                            arrowprops=dict(arrowstyle="-|>", color="#a63c1e", lw=1.5))

    # ── Final emission arrow at y = M+1 (always ket-side deexcitation) ────
    # Tail at (x_ket, M+1), tip at (x_out_ket, M+1.5): outward & upward.
    ax.annotate("", xy=(x_out_ket, M + 1.5), xytext=(x_ket, M + 1),
                arrowprops=dict(arrowstyle="-|>", color="#1a6faf", lw=1.5))

    # ── Waiting-time labels t_k ───────────────────────────────────────────
    # t_k is the duration of the interval between arrows k and k+1
    # (or between arrow M and the final emission arrow).
    # Its label sits midway between dashed lines k and k+1, i.e. at y = k + 0.5,
    # placed just to the right of the ket line so it never overlaps an arrow.
    for k in range(1, M + 1):
        ax.text(x_ket + 0.08, k + 0.5, f"$t_{k}$",
                ha="left", va="center", fontsize=10, color="black")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Block 1] Feynman diagram saved to: {outfile}")


def print_response_function_expression(params, e_seq, tau_coeffs, outfile_prefix="response"):
    """
    Block 2: Build a LaTeX document with the fully-substituted response function
    equations (Eqs. 3-11 of HT.pdf), compile it to PDF.

    Substituted:  e_k, ket_k, bra_k indices; fixed t values inside chi expressions.
    Kept symbolic: varepsilon_j, omega_m, z_j^(m), tau_dep, tau_rel, mu_{jl}, hbar, k_B, T.
    """
    M         = params["M"]
    kets      = params["kets"]
    bras      = params["bras"]
    x_val     = compute_x(params)
    K         = len(e_seq)
    Nm        = len(params["omega_vib"])
    fixed     = params["fixed_times"]
    mu0       = params["mu_0"]
    mu1_list  = params.get("mu1")
    mu2_list  = params.get("mu2")
    use_HT1   = params.get("use_HT1", False)
    use_HT2   = params.get("use_HT2", False)
    use_HT3   = params.get("use_HT3", False)
    e_full    = [0] + list(e_seq) + [0] * (M - K + 1)

    print("\n" + "=" * 70)
    print("  BLOCK 2 — Response Function Expression (LaTeX)")
    print("=" * 70)
    print(f"\n  Order M = {M},  x = {x_val},  K = {K}")
    print(f"  Electronic state sequence:  " +
          "  ".join([f"e_0=0"] +
                    [f"e_{k+1}={e}" for k, e in enumerate(e_seq)] +
                    [f"e_{{K+1}}=0"]))

    # ── Helper: tau_k as LaTeX signed sum of t_l ──────────────────────────────
    def tau_latex(coeffs, parens=False):
        if not coeffs:
            return "0"
        terms = []
        for l in sorted(coeffs.keys()):
            c = coeffs[l]
            if   c ==  1: terms.append(f"t_{l}")
            elif c == -1: terms.append(f"-t_{l}")
            elif c  >  0: terms.append(f"{c}\\,t_{l}")
            else:         terms.append(f"{c:g}\\,t_{l}")
        result = terms[0]
        for t in terms[1:]:
            result += t if t.startswith("-") else "+" + t
        if parens and (len(terms) > 1 or terms[0].startswith("-")):
            return "(" + result + ")"
        return result

    # ── Helper: chi^(m)_{k,l} as LaTeX, substituting fixed t values ──────────
    def chi_arg_latex(k, l):
        """Argument of exp in chi^(m)_{k,l} = exp(-i omega_m (t_k+...+t_l)).
        Sums raw t_k..t_l directly (chi uses t, not tau).
        Fixed t values substituted numerically. Returns None if arg=0.
        Indices k,l refer to the t_1..t_M numbering (1-based).
        """
        if l < k:
            return None   # chi_{k,k-1} = 1 by convention
        numeric = sum(fixed[p] for p in range(k, l + 1) if p in fixed)
        symbolic = [p for p in range(k, l + 1) if p not in fixed]
        if numeric == 0 and not symbolic:
            return None
        terms = []
        if numeric != 0:
            terms.append(f"{numeric:g}")
        for p in symbolic:
            terms.append(f"t_{p}")
        arg = terms[0]
        for t in terms[1:]:
            arg += "+" + t
        return arg

    def tau_arg_latex(l, k):
        """Argument of exp in f_m product: sum_{p=l}^{l+k-1} tau_p,
        with tau_p expressed via tau_coeffs and fixed t's substituted.
        Returns None if arg=0.
        """
        combined = {}
        for p in range(l, l + k):
            for tidx, c in tau_coeffs[p - 1].items():
                combined[tidx] = combined.get(tidx, 0) + c
        combined = {i: c for i, c in combined.items() if c != 0}
        if not combined:
            return None
        numeric = sum(c * fixed[tidx] for tidx, c in combined.items() if tidx in fixed)
        symbolic = {tidx: c for tidx, c in combined.items() if tidx not in fixed}
        terms = []
        if numeric != 0:
            terms.append(f"{numeric:g}")
        for tidx in sorted(symbolic.keys()):
            c = symbolic[tidx]
            if   c ==  1: terms.append(f"t_{tidx}")
            elif c == -1: terms.append(f"-t_{tidx}")
            else:         terms.append(f"{c:g}\\,t_{tidx}")
        if not terms:
            return None
        arg = terms[0]
        for t in terms[1:]:
            arg += t if t.startswith("-") else "+" + t
        return arg

    def chi_latex(k, l):
        """LaTeX for chi^(m)_{k,l}."""
        arg = chi_arg_latex(k, l)
        if arg is None:
            return "1"
        return f"e^{{-i\\omega_m({arg})}}"

    def chi_conj(k, l):
        """LaTeX for (chi^(m)_{k,l})^* — flips sign of exponent."""
        arg = chi_arg_latex(k, l)
        if arg is None:
            return "1"
        # negate all coefficients
        return f"e^{{i\\omega_m({arg})}}"

    # ── Helper: a_k closed form ───────────────────────────────────────────────
    # a_k = sum_{j=1}^{k-1} z_{e_j} (chi_{j,j} - 1) chi_{j+1,k-1}
    def a_latex(k):
        terms = []
        for j in range(1, k):
            chi_jj  = chi_latex(j, j)
            if chi_jj == "1":
                continue   # (chi-1)=0
            chi_prod = chi_latex(j + 1, k - 1)
            z_ej = f"z_{{{e_full[j]}}}^{{(m)}}"
            term = f"{z_ej}({chi_jj}-1)"
            if chi_prod != "1":
                term += chi_prod
            terms.append(term)
        if not terms:
            return "0"
        result = terms[0]
        for t in terms[1:]:
            result += "+" + t
        return result

    # ── Helper: c_k closed form ───────────────────────────────────────────────
    # c_k = -sum_{j=k}^{K} z_{e_j} (chi_{j,j} - 1) chi_{k,j-1}
    def c_latex(k):
        terms = []
        for j in range(k, K + 1):
            chi_jj  = chi_latex(j, j)
            if chi_jj == "1":
                continue
            chi_prod = chi_latex(k, j - 1)
            z_ej = f"z_{{{e_full[j]}}}^{{(m)}}"
            term = f"{z_ej}({chi_jj}-1)"
            if chi_prod != "1":
                term += chi_prod
            terms.append(term)
        if not terms:
            return "0"
        inner = terms[0]
        for t in terms[1:]:
            inner += "+" + t
        return f"-({inner})"

    # ── Helper: (a_k + c_k) simplified ───────────────────────────────────────
    def ac_latex(k):
        a = a_latex(k)
        c = c_latex(k)
        if a == "0" and c == "0":
            return "0"
        if a == "0":
            return c
        if c == "0":
            return a
        return f"{a}+{c}"

    # ── Helper: chi^(m)_{1,k-1} ───────────────────────────────────────────────
    def chi1_latex(k):
        return chi_latex(1, k - 1)

    # ── Helper: C and its variants ────────────────────────────────────────────
    def C_latex():
        factors = [f"\\mu^{{(0)}}_{{{e_full[k]}{e_full[k+1]}}}" for k in range(K + 1)]
        return "".join(factors)

    def C_mk_latex(m_idx, k):
        """C^(m)_{k|} = C * mu1[m][e_{k-1},e_k] / mu0[e_{k-1},e_k]"""
        j1, j2 = e_full[k - 1], e_full[k]
        denom = mu0[j1, j2]
        if denom == 0 or mu1_list is None:
            return None
        mu1_val = mu1_list[m_idx][j1, j2]
        if mu1_val == 0:
            return None
        ratio = f"\\frac{{\\mu^{{({m_idx+1})}}_{{{j1}{j2}}}}}{{\\mu^{{(0)}}_{{{j1}{j2}}}}}"
        return f"C\\,{ratio}"

    def C_mn_kl_latex(m_idx, n_idx, k, l):
        """C^(mn)_{kl|}: only nonzero if both mu1 elements are nonzero."""
        j1k, j2k = e_full[k - 1], e_full[k]
        j1l, j2l = e_full[l - 1], e_full[l]
        if mu0[j1k, j2k] == 0 or mu0[j1l, j2l] == 0 or mu1_list is None:
            return None
        if mu1_list[m_idx][j1k, j2k] == 0 or mu1_list[n_idx][j1l, j2l] == 0:
            return None
        r1 = f"\\frac{{\\mu^{{({m_idx+1})}}_{{{j1k}{j2k}}}}}{{\\mu^{{(0)}}_{{{j1k}{j2k}}}}}"
        r2 = f"\\frac{{\\mu^{{({n_idx+1})}}_{{{j1l}{j2l}}}}}{{\\mu^{{(0)}}_{{{j1l}{j2l}}}}}"
        return f"C\\,{r1}\\,{r2}"

    def C_mk2_latex(m_idx, k):
        """C^(m)_{|k} = C * mu2[m][e_{k-1},e_k] / mu0[e_{k-1},e_k]"""
        j1, j2 = e_full[k - 1], e_full[k]
        denom = mu0[j1, j2]
        if denom == 0 or mu2_list is None:
            return None
        mu2_val = mu2_list[m_idx][j1, j2]
        if mu2_val == 0:
            return None
        ratio = f"\\frac{{\\tilde{{\\mu}}^{{({m_idx+1})}}_{{{j1}{j2}}}}}{{\\mu^{{(0)}}_{{{j1}{j2}}}}}"
        return f"C\\,{ratio}"

    # ── Helper: I^(m)_{ab} and J^(m)_{11} symbolic ───────────────────────────
    def I_sym(a, b):
        return f"I^{{(m)}}_{{\\!{a}{b}}}"
    J_sym = r"J^{(m)}_{11}"

    # ── Helper: bracket for R1/R2 at position k ───────────────────────────────
    def bracket_k(k):
        """[I00*(a_k+c_k) + I10*chi1_{k-1} + I01*chi1_{k-1}^*]"""
        ac = ac_latex(k)
        ch = chi1_latex(k)
        parts = []
        if ac != "0":
            parts.append(f"{I_sym(0,0)}({ac})")
        if ch != "1":
            parts.append(f"{I_sym(1,0)}{ch}")
            parts.append(f"{I_sym(0,1)}{chi_conj(1,k-1)}")
        elif ch == "1":   # chi1=1, still include
            parts.append(f"{I_sym(1,0)}")
            parts.append(f"{I_sym(0,1)}")
        if not parts:
            return "0"
        return "+".join(parts)

    # ── tau_lines for preamble ─────────────────────────────────────────────────
    tau_lines = "\n".join(
        f"  \\tau_{k+1} &= {tau_latex(tau_coeffs[k])} \\\\"
        for k in range(K)
    )

    # ── Oscillatory exponent (Eq. 11) ─────────────────────────────────────────
    osc_terms = []
    for k in range(1, M + 1):
        dk = kets[k] - bras[k]
        if dk == 0:
            continue
        diff_str = f"(\\varepsilon_{{{kets[k]}}}-\\varepsilon_{{{bras[k]}}})"
        # substitute fixed t
        if k in fixed:
            val = fixed[k] * dk
            osc_terms.append(f"{val:g}\\,{diff_str}")
        else:
            prefix = "" if dk == 1 else (f"-" if dk == -1 else f"{dk:g}\\,")
            osc_terms.append(f"{'+' if osc_terms and dk>0 else ''}t_{k}\\,{diff_str}")
    osc_latex = "".join(osc_terms) if osc_terms else "0"

    # ── Decay exponent (Eq. 11) ───────────────────────────────────────────────
    decay_terms = []
    for k in range(1, M + 1):
        tk = f"{fixed[k]:g}" if k in fixed else f"t_{k}"
        if bras[k] != kets[k]:
            decay_terms.append(f"\\dfrac{{{tk}}}{{\\tau_{{\\mathrm{{dep}}}}}}")
        elif kets[k] != 0:
            decay_terms.append(f"\\dfrac{{{tk}}}{{\\tau_{{\\mathrm{{rel}}}}}}")
    decay_latex = "+".join(decay_terms) if decay_terms else "0"

    # ── C value and Eq. 5 ────────────────────────────────────────────────────
    C_val_latex = C_latex()

    # ── Build R0 (Eq. 7) ─────────────────────────────────────────────────────
    prod_I00 = "".join(f"I^{{(m)}}_{{{m+1},00}}" if Nm > 1
                       else "I^{{(m)}}_{{00}}" for m in range(Nm))
    R0_latex = f"C\\,F\\,{prod_I00 if Nm <= 3 else r'\prod_{m=1}^{N_m}I^{(m)}_{00}'}"

    # ── Build R1 (Eq. 8) ─────────────────────────────────────────────────────
    R1_lines = []
    if use_HT1 and mu1_list is not None:
        for k in range(1, K + 2):
            for m_idx in range(Nm):
                C_mk = C_mk_latex(m_idx, k)
                if C_mk is None:
                    continue
                bk = bracket_k(k)
                # product of I00 over other modes
                other = "".join(f"I^{{({mm+1})}}_{{{mm+1},00}}"
                                for mm in range(Nm) if mm != m_idx) if Nm > 1 else ""
                line = f"  {C_mk}\\,F\\,{bk}"
                if other:
                    line += f"\\,{other}"
                R1_lines.append(line)

    # ── Build R2 (Eq. 9) ─────────────────────────────────────────────────────
    R2_lines = []
    if use_HT2 and mu1_list is not None:
        # First double sum
        for k in range(1, K + 2):
            for l in range(k + 1, K + 2):
                for m_idx in range(Nm):
                    for n_idx in range(Nm):
                        C_mn = C_mn_kl_latex(m_idx, n_idx, k, l)
                        if C_mn is None:
                            continue
                        bk = bracket_k(k)
                        bl = bracket_k(l)
                        # product over modes other than m and n
                        excl = {m_idx, n_idx}
                        other = "".join(f"I^{{({mm+1})}}_{{{mm+1},00}}"
                                        for mm in range(Nm) if mm not in excl) if Nm > 1 else ""
                        line = f"  {C_mn}\\,F\\,({bk})({bl})"
                        if other:
                            line += f"\\,{other}"
                        R2_lines.append(line)
        # Second sum (same-mode cross terms)
        for k in range(1, K + 2):
            for l in range(k + 1, K + 2):
                j1k, j2k = e_full[k-1], e_full[k]
                j1l, j2l = e_full[l-1], e_full[l]
                if mu0[j1k, j2k] == 0 or mu0[j1l, j2l] == 0:
                    continue
                for m_idx in range(Nm):
                    if mu1_list[m_idx][j1k, j2k] == 0 or mu1_list[m_idx][j1l, j2l] == 0:
                        continue
                    r1 = f"\\frac{{\\mu^{{({m_idx+1})}}_{{{j1k}{j2k}}}}}{{\\mu^{{(0)}}_{{{j1k}{j2k}}}}}"
                    r2 = f"\\frac{{\\mu^{{({m_idx+1})}}_{{{j1l}{j2l}}}}}{{\\mu^{{(0)}}_{{{j1l}{j2l}}}}}"
                    C_mm = f"C\\,{r1}\\,{r2}"
                    chi1k = chi1_latex(k)
                    chi1l = chi1_latex(l)
                    chi_kl = chi_latex(k, l - 1)
                    other = "".join(f"I^{{({mm+1})}}_{{{mm+1},00}}"
                                    for mm in range(Nm) if mm != m_idx) if Nm > 1 else ""
                    chi1k_c = chi_conj(1, k-1)
                    chi1l_c = chi_conj(1, l-1)
                    bk_I = f"({I_sym(1,0)}{chi1k}+{I_sym(0,1)}{chi1k_c})"
                    bl_I = f"({I_sym(1,0)}{chi1l}+{I_sym(0,1)}{chi1l_c})"
                    inner = (f"{I_sym(2,0)}{chi1k}{chi1l}"
                             f"+{I_sym(0,2)}{chi1k_c}{chi1l_c}"
                             f"+{J_sym}({chi1k}{chi1l_c}+{chi1l}{chi1k_c})"
                             f"+{I_sym(0,0)}{chi_kl}"
                             f"-{bk_I}{bl_I}")
                    line = f"  {C_mm}\\,F\\,\\bigl({inner}\\bigr)"
                    if other:
                        line += f"\\,{other}"
                    R2_lines.append(line)

    # ── Build R3 (Eq. 10) ────────────────────────────────────────────────────
    R3_lines = []
    if use_HT3 and mu2_list is not None:
        for k in range(1, K + 2):
            for m_idx in range(Nm):
                C_mk2 = C_mk2_latex(m_idx, k)
                if C_mk2 is None:
                    continue
                ac = ac_latex(k)
                chi1k = chi1_latex(k)
                other = "".join(f"I^{{({mm+1})}}_{{{mm+1},00}}"
                                for mm in range(Nm) if mm != m_idx) if Nm > 1 else ""
                inner_parts = [I_sym(0, 0)]
                if ac != "0":
                    inner_parts.append(f"{I_sym(0,0)}({ac})^2")
                    inner_parts.append(f"({I_sym(1,0)}+{I_sym(0,1)})({ac})")
                inner_parts.append(f"{I_sym(2,0)}{chi1k}^2")
                chi1k_c = chi_conj(1, k-1)
                inner_parts.append(f"{I_sym(0,2)}{chi1k}{chi1k_c}")
                inner_parts.append(f"2{I_sym(1,1)}")
                inner = "+".join(inner_parts)
                line = f"  {C_mk2}\\,F\\,\\bigl({inner}\\bigr)"
                if other:
                    line += f"\\,{other}"
                R3_lines.append(line)

    # ── Build LaTeX document ──────────────────────────────────────────────────
    def lines_to_eq(lines, label):
        if not lines:
            return f"\\begin{{equation}}\n{label} = 0\n\\end{{equation}}\n"
        joined = " \\\\\n  &+".join(lines)
        return (f"\\begin{{align}}\n{label} &= {lines[0]}"
                + (" \\\\\n  &+".join([""] + lines[1:]) if len(lines) > 1 else "")
                + "\n\\end{align}\n")

    def simple_eq(label, rhs):
        return f"\\begin{{equation}}\n{label} = {rhs}\n\\end{{equation}}\n"

    # Section: whether HT terms are included
    ht_note = ""
    flags = []
    if use_HT1: flags.append(r"$R^{(v,M)}_{T,1}$ (linear in $\hat{\mu}_1$)")
    if use_HT2: flags.append(r"$R^{(v,M)}_{T,2}$ (quadratic in $\hat{\mu}_1$)")
    if use_HT3: flags.append(r"$R^{(v,M)}_{T,3}$ (linear in $\hat{\mu}_2$)")
    if flags:
        ht_note = "Including HT contributions: " + ", ".join(flags) + ".\n\n"
    else:
        ht_note = r"Only Franck-Condon contribution ($R^{(v,M)}_{T,0}$) included." + "\n\n"

    doc = (r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\usepackage[margin=2cm]{geometry}
\begin{document}

\begin{center}
{\large\bfseries Response Function $R^{(""" + str(M) + r""")}$ --- Equations}
\end{center}

\medskip
\noindent\textbf{Diagram parameters:}
$M=""" + str(M) + r"""$, $x=""" + str(x_val) + r"""$, $K=""" + str(K) + r"""$.\\
Electronic state sequence: $e_0=0""" +
"".join(f",\\;e_{{{k+1}}}={e}" for k, e in enumerate(e_seq)) +
r""",\;e_{K+1}=0$.\\
""" + (f"Fixed times: " +
       ", ".join(f"$t_{k}={v:g}$~fs" for k, v in sorted(fixed.items())) + ".\\\\") +
r"""
Free time axes: """ +
", ".join(f"$t_{k}$" for k in range(1, M+1) if k not in fixed) + r""".

\medskip
\noindent """ + ht_note +
r"""
\noindent\textbf{Intervals $\tau_k$ (in terms of free time variables):}
\begin{align*}
""" + tau_lines + r"""
\end{align*}

%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(3): $I^{(m)}_{ab}$ and $J^{(m)}_{11}$}

\begin{align}
I^{(m)}_{ab} &= \langle n_m\rangle^{2(a+b)}\,(Q_m^*)^a Q_m^b\,
  e^{-\langle n_m\rangle|Q_m|^2} \\
J^{(m)}_{11} &= \langle n_m\rangle^2\bigl[1+\langle n_m\rangle|Q_m|^2\bigr]
  e^{-\langle n_m\rangle|Q_m|^2}
\end{align}
where $\langle n_m\rangle = 1/(e^{\hbar\omega_m/k_{\rm B}T}-1)$.

%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(4): $Q_m$}

\begin{equation}
Q_m = 2\sum_{j=1}^{K} z_{e_j}^{(m)}\bigl(\chi^{(m)*}_{j,j}-1\bigr)\,\chi^{(m)*}_{1,j-1}
\end{equation}
where $\chi^{(m)}_{k,l}=e^{-i\omega_m(t_k+\cdots+t_l)}$
(and $\chi^{(m)}_{k,k-1}=1$). The diagonal factors with fixed $t$-values substituted:
\begin{align*}
""" +
"\n".join(
    f"  \\chi^{{(m)}}_{{{k},{k}}} &= {chi_latex(k,k)} \\\\"
    for k in range(1, K+1)
) +
r"""
\end{align*}
Substituting $e_j$ values and fixed $t$'s (2 terms per line):
\begin{align*}
  Q_m &= 2\Bigl[""" +
(" \\\\\n  &\\qquad +".join(
    f"z_{{{e_full[j]}}}^{{(m)}}\\bigl({chi_conj(j,j)}-1\\bigr)"
    + (chi_conj(1,j-1) if chi_conj(1,j-1) != "1" else "")
    for j in range(1, K+1)
    if chi_conj(j,j) != "1"   # skip terms where chi=1 → (1-1)=0
) or "0") +
r"""\Bigr]
\end{align*}

%% ─────────────────────────────────────────────────────────────────
\section*{$f_m$ functions}

The lineshape functions $f_m$ entering $F=\exp\!\bigl(\sum_m f_m\bigr)$,
with fixed $t$-values substituted:
\begin{align*}
  f_m &= """ +
(" \\\\\n  &\\quad +".join(
    (lambda z1, z2, ep:
        f"{z1}{z2}\\bigl(1-{ep}\\bigr)"
    )(
        (f"[-z_{{{e_full[l]}}}^{{(m)}}]" if e_full[l-1]==0 else
         f"z_{{{e_full[l-1]}}}^{{(m)}}" if e_full[l]==0 else
         f"(z_{{{e_full[l-1]}}}^{{(m)}}-z_{{{e_full[l]}}}^{{(m)}})"),
        (f"[-z_{{{e_full[l+kk]}}}^{{(m)}}]" if e_full[l+kk-1]==0 else
         f"z_{{{e_full[l+kk-1]}}}^{{(m)}}" if e_full[l+kk]==0 else
         f"(z_{{{e_full[l+kk-1]}}}^{{(m)}}-z_{{{e_full[l+kk]}}}^{{(m)}})"),
        (lambda arg: "1" if arg is None else f"e^{{-i\\omega_m({arg})}}")(tau_arg_latex(l, kk))
    )
    for kk in range(1, K+1)
    for l in range(1, K-kk+2)
    if not (e_full[l-1]==e_full[l] or e_full[l+kk-1]==e_full[l+kk])
    and tau_arg_latex(l, kk) is not None
) or "0") +
r"""
\end{align*}

%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(5): $C$ and its variants}

\begin{equation}
C = """ + C_val_latex + r"""
\end{equation}
where $\mu^{(0)}_{jl}\equiv\langle j|\hat{\mu}_0|l\rangle$.
""" +
(r"""The non-zero $C^{(m)}_{k|}$ coefficients (for the active $(k,m)$ pairs) are
defined by $C^{(m)}_{k|}=C\,\mu^{(m)}_{e_{k-1}e_k}/\mu^{(0)}_{e_{k-1}e_k}$,
where $\mu^{(m)}_{jl}\equiv\langle j|\hat{\mu}_1^{(m)}|l\rangle$.
""" if use_HT1 or use_HT2 else "") +
(r"""The $C^{(m)}_{|k}$ coefficients are defined by
$C^{(m)}_{|k}=C\,\tilde{\mu}^{(m)}_{e_{k-1}e_k}/\mu^{(0)}_{e_{k-1}e_k}$,
where $\tilde{\mu}^{(m)}_{jl}\equiv\langle j|\hat{\mu}_2^{(m)}|l\rangle$.
""" if use_HT3 else "") +
r"""
%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(6): Vibrational response function}

\begin{equation}
R^{(v,M)}_T = R^{(v,M)}_{T,0}""" +
(r" + R^{(v,M)}_{T,1}" if use_HT1 else "") +
(r" + R^{(v,M)}_{T,2}" if use_HT2 else "") +
(r" + R^{(v,M)}_{T,3}" if use_HT3 else "") +
r"""
\end{equation}
with $F = \exp\!\bigl(\sum_{m=1}^{N_m} f_m\bigr)$.

%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(7): Franck-Condon contribution $R^{(v,M)}_{T,0}$}

""" + simple_eq(r"R^{(v,M)}_{T,0}", R0_latex) +
(r"""
%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(8): HT contribution $R^{(v,M)}_{T,1}$ (linear in $\hat{\mu}_1$)}

""" + (lines_to_eq(R1_lines, r"R^{(v,M)}_{T,1}") if R1_lines
       else r"\begin{equation}R^{(v,M)}_{T,1}=0\text{ (all }C^{(m)}_{k|}=0)\end{equation}" + "\n")
    if use_HT1 else "") +
(r"""
%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(9): HT contribution $R^{(v,M)}_{T,2}$ (quadratic in $\hat{\mu}_1$)}

""" + (lines_to_eq(R2_lines, r"R^{(v,M)}_{T,2}") if R2_lines
       else r"\begin{equation}R^{(v,M)}_{T,2}=0\text{ (all }C^{(mn)}_{kl|}=0)\end{equation}" + "\n")
    if use_HT2 else "") +
(r"""
%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(10): HT contribution $R^{(v,M)}_{T,3}$ (linear in $\hat{\mu}_2$)}

""" + (lines_to_eq(R3_lines, r"R^{(v,M)}_{T,3}") if R3_lines
       else r"\begin{equation}R^{(v,M)}_{T,3}=0\text{ (all }C^{(m)}_{|k}=0)\end{equation}" + "\n")
    if use_HT3 else "") +
r"""
%% ─────────────────────────────────────────────────────────────────
\section*{Eq.~(11): Full response function}

\begin{equation}
R^{(M)}_T = R^{(v,M)}_T \times (-1)^{""" + str(x_val) + r"""}
  \exp\!\left[-\frac{i}{\hbar}\Bigl(""" + osc_latex + r"""\Bigr)\right]
  \exp\!\left[-\Bigl(""" + decay_latex + r"""\Bigr)\right]
\end{equation}

\end{document}
""")

    tex_file = outfile_prefix + "_equations.tex"
    pdf_file = outfile_prefix + "_equations.pdf"
    with open(tex_file, "w") as f:
        f.write(doc)
    print(f"\n  [Block 2] LaTeX source written to: {tex_file}")

    import subprocess, os
    if params.get("compile_latex", True):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory",
             os.path.dirname(os.path.abspath(tex_file)), tex_file],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  [Block 2] PDF compiled:           {pdf_file}")
        else:
            print(f"  [Block 2] pdflatex failed. Check {tex_file}")
            # Show last 30 lines of log for diagnosis
            lines = result.stdout.splitlines()
            print("\n".join(lines[-30:]))
    else:
        print(f"  [Block 2] LaTeX compilation skipped (compile_latex = false)")
        pdf_file = None

    return tex_file, pdf_file



# ══════════════════════════════════════════════════════════════════════════════
#  BLOCKS 4–5 — Herzberg-Teller contributions
# ══════════════════════════════════════════════════════════════════════════════

def chi(m_idx, k, l, taus, omega_vib):
    """
    χ^(m)_{k,l} = exp(-i * omega_m * sum_{p=k}^{l} tau_p)  for k <= l
    χ^(m)_{k,k-1} = 1  (empty product convention)
    taus: array of K tau values (0-indexed: taus[p-1] = tau_p)
    """
    if l < k:
        return 1.0 + 0.0j
    om = omega_vib[m_idx] / HBAR_meV_fs   # meV → rad/fs
    phase = om * sum(taus[p - 1] for p in range(k, l + 1))
    return np.exp(-1j * phase)


def compute_a_c(m_idx, e_seq, taus, z, omega_vib):
    """
    Recursive computation of a^(M)_{k,m} (k=1..M+1) and c^(M)_{k,m} (k=1..M+1).
    Eq. (1): a_{k,m} = a_{k-1,m} * chi_{k-1,k-1} + z_{e_{k-1}} * (chi_{k-1,k-1} - 1)
             a_{1,m} = 0
    Eq. (2): c_{k,m} = c_{k+1,m} * chi_{k,k} - z_{e_k} * (chi_{k,k} - 1)
             c_{M+1,m} = 0
    e_seq: length K sequence; e_0=e_{K+1}=0 appended.
    Note: in the paper k runs 1..M+1 but e_k is only defined for k=0..K (with K<=M).
          We use e_full[k] which maps k -> e_k with proper padding.
    """
    K = len(e_seq)
    M_eff = K       # the recursion runs over M+1 terms; paper's M = K here
    e_full = [0] + list(e_seq) + [0] * (M_eff)   # e_full[k] = e_k, k=0..M+1

    # Forward: a[k] for k=1..M+1  (0-indexed: a[0]=a_1=0)
    a = np.zeros(M_eff + 2, dtype=complex)
    for k in range(2, M_eff + 2):   # k=2..M+1
        chi_prev = chi(m_idx, k-1, k-1, taus, omega_vib)  # chi_{k-1,k-1}
        z_ek_prev = z[e_full[k-1], m_idx]
        a[k] = a[k-1] * chi_prev + z_ek_prev * (chi_prev - 1.0)

    # Backward: c[k] for k=M+1..1  (0-indexed: c[M+1]=c_{M+2}=0, boundary c_{M+1}=0)
    c = np.zeros(M_eff + 2, dtype=complex)
    for k in range(M_eff, 0, -1):   # k=M..1
        chi_kk = chi(m_idx, k, k, taus, omega_vib)        # chi_{k,k}
        z_ek = z[e_full[k], m_idx]
        c[k] = c[k+1] * chi_kk - z_ek * (chi_kk - 1.0)

    return a, c   # 1-indexed: a[k] = a_k, c[k] = c_k, k=1..M+1


def compute_Q(m_idx, e_seq, taus, z, omega_vib):
    """
    Q_m = 2 * sum_{j=1}^{M} z_{e_j}^(m) * (chi_{j,j}^* - 1) * chi_{1,j-1}^*
    Eq. (4).
    """
    K = len(e_seq)
    e_full = [0] + list(e_seq) + [0] * K

    Q = 0.0 + 0.0j
    for j in range(1, K + 1):
        chi_jj    = chi(m_idx, j, j, taus, omega_vib)
        chi_1_jm1 = chi(m_idx, 1, j-1, taus, omega_vib)   # =1 when j=1
        z_ej = z[e_full[j], m_idx]
        Q += z_ej * (np.conj(chi_jj) - 1.0) * np.conj(chi_1_jm1)
    return 2.0 * Q


def bose(omega_m_meV, T):
    """Mean occupation <n_m> = 1/(exp(hbar*omega_m / k_B T) - 1)."""
    if T <= 0:
        return 0.0
    x = omega_m_meV / (KB_meV_K * T)
    if x > 500:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)


def I_ab(a, b, Q, n_m):
    """
    I^(m)_{ab} = <n_m>^{2(a+b)} * (Q^*)^a * Q^b * exp(-<n_m>|Q|^2)
    Eq. (3).
    At T=0 (n_m=0): I_00 = 1 exactly; all other I_ab = 0.
    """
    if n_m == 0.0:
        return 1.0 + 0.0j if (a == 0 and b == 0) else 0.0 + 0.0j
    return (n_m ** (2 * (a + b))
            * (np.conj(Q) ** a) * (Q ** b)
            * np.exp(-n_m * abs(Q)**2))


def J_11(Q, n_m):
    """
    J^(m)_{11} = <n_m>^2 * [1 + <n_m>|Q|^2] * exp(-<n_m>|Q|^2)
    Eq. (3).
    """
    return n_m**2 * (1.0 + n_m * abs(Q)**2) * np.exp(-n_m * abs(Q)**2)


def compute_rvib_HT(m_idx, e_seq, taus, z, omega_vib, T,
                    mu0, mu1_list, mu2_list,
                    use_HT1, use_HT2, use_HT3):
    """
    Compute R^(v,M)_T contributions for all HT terms.

    Returns a dict with keys 'R0', 'R1', 'R2', 'R3' (complex scalars).
    Each is the vibrational factor for the corresponding term, *before*
    multiplication by the electronic component and F = exp(sum_m f_m).
    (F is applied once in compute_response_HT after summing over modes.)

    Actually per Eqs (7-10) the sums over m, n run inside R1,R2,R3 while
    R0 is a product over modes. We compute the full sums here.
    """
    pass   # see compute_response_HT below which handles all modes together


def compute_R_vib_full(e_seq, taus, z, omega_vib, T,
                       mu0, mu1_list, mu2_list,
                       use_HT1, use_HT2, use_HT3):
    """
    Compute the full vibrational response function R^(v,M)_T = R0 + R1 + R2 + R3
    following Eqs. (6-11) of HT.pdf.

    Parameters
    ----------
    e_seq    : list of ints, length K
    taus     : array of K tau values (fs)
    z        : (N_elec, Nm) displacement array
    omega_vib: (Nm,) array of vibrational frequencies in meV
    T        : temperature in K
    mu0      : (N_elec, N_elec) array  — permanent dipole (mu_0)
    mu1_list : list of Nm (N_elec, N_elec) arrays, or None
    mu2_list : list of Nm (N_elec, N_elec) arrays, or None
    use_HT1/2/3 : bool flags

    Returns
    -------
    R_vib : complex scalar = R^(v,M)_T  (to be multiplied by R^(e,M))
    """
    K  = len(e_seq)
    Nm = len(omega_vib)
    e_full = [0] + list(e_seq) + [0] * K   # e_full[k] for k=0..K+1

    # ── C = prod_{k=0}^{K} mu0[e_k, e_{k+1}] ─────────────────────────────
    C = 1.0
    for k in range(K + 1):
        C *= mu0[e_full[k], e_full[k + 1]]

    # ── Per-mode quantities ────────────────────────────────────────────────
    fm_arr   = np.zeros(Nm, dtype=complex)
    n_arr    = np.zeros(Nm)
    Q_arr    = np.zeros(Nm, dtype=complex)
    a_arr    = []    # a_arr[m] = array a[k], k=1..K+1
    c_arr    = []    # c_arr[m] = array c[k]
    I00_arr  = np.zeros(Nm, dtype=complex)
    chi1_arr = []    # chi1_arr[m][k] = chi^(m)_{1,k-1}  for k=1..K+1

    for m in range(Nm):
        # f_m (Franck-Condon, from compute_fm)
        from itertools import product as iproduct
        fm_arr[m] = compute_fm(m, taus, e_seq, z, omega_vib)

        # <n_m>
        n_arr[m] = bose(omega_vib[m], T)

        # Q_m
        Q_arr[m] = compute_Q(m, e_seq, taus, z, omega_vib)

        # a, c
        a_m, c_m = compute_a_c(m, e_seq, taus, z, omega_vib)
        a_arr.append(a_m)
        c_arr.append(c_m)

        # I^(m)_{00}
        I00_arr[m] = I_ab(0, 0, Q_arr[m], n_arr[m])

        # chi^(m)_{1,k-1} for k=1..K+1
        chi1_k = np.zeros(K + 2, dtype=complex)
        for k in range(1, K + 2):
            chi1_k[k] = chi(m, 1, k-1, taus, omega_vib)
        chi1_arr.append(chi1_k)

    # ── F = exp(sum_m f_m) ─────────────────────────────────────────────────
    F = np.exp(np.sum(fm_arr))

    # ── R0: Franck-Condon term (Eq. 7) ────────────────────────────────────
    # R^(v,M)_{T,0} = C * F * prod_m I^(m)_{00}
    R0 = C * F * np.prod(I00_arr)

    # ── R1: linear in mu1 (Eq. 8) ─────────────────────────────────────────
    R1 = 0.0 + 0.0j
    if use_HT1 and mu1_list is not None:
        for k in range(1, K + 2):   # k=1..M+1
            # denom = mu0[e_{k-1}, e_k]
            denom_k = mu0[e_full[k-1], e_full[k]]
            for m in range(Nm):
                if denom_k == 0.0:
                    continue
                # C^(m)_{k|} = C * mu1[m][e_{k-1}, e_k] / mu0[e_{k-1}, e_k]
                C_mk = C * mu1_list[m][e_full[k-1], e_full[k]] / denom_k

                I00 = I00_arr[m]
                I10 = I_ab(1, 0, Q_arr[m], n_arr[m])
                I01 = I_ab(0, 1, Q_arr[m], n_arr[m])
                chi1k = chi1_arr[m][k]

                # Product of I00 over all OTHER modes
                prod_other = np.prod([I00_arr[mm] for mm in range(Nm) if mm != m])

                bracket = (I00 * (a_arr[m][k] + c_arr[m][k])
                           + I10 * chi1k
                           + I01 * np.conj(chi1k))

                R1 += C_mk * F * bracket * prod_other

    # ── R2: quadratic in mu1 (Eq. 9) ──────────────────────────────────────
    R2 = 0.0 + 0.0j
    if use_HT2 and mu1_list is not None:
        # First sum: k<l, modes m,n (can be same mode)
        for k in range(1, K + 2):
            denom_k = mu0[e_full[k-1], e_full[k]]
            for l in range(k + 1, K + 2):
                denom_l = mu0[e_full[l-1], e_full[l]]
                for m in range(Nm):
                    if denom_k == 0.0:
                        continue
                    C_mk = C * mu1_list[m][e_full[k-1], e_full[k]] / denom_k
                    I00m = I00_arr[m]
                    I10m = I_ab(1, 0, Q_arr[m], n_arr[m])
                    I01m = I_ab(0, 1, Q_arr[m], n_arr[m])
                    chi1k_m = chi1_arr[m][k]
                    brack_k_m = (I00m * (a_arr[m][k] + c_arr[m][k])
                                 + I10m * chi1k_m
                                 + I01m * np.conj(chi1k_m))

                    for n in range(Nm):
                        if denom_l == 0.0:
                            continue
                        C_nl = C * mu1_list[n][e_full[l-1], e_full[l]] / denom_l
                        # combined C^(mn)_{kl|} = C^(m)_{k|} * mu1[n][e_{l-1},e_l] / mu0[e_{l-1},e_l]
                        # but C^(m)_{k|} already contains C, so:
                        C_mn_kl = C_mk * mu1_list[n][e_full[l-1], e_full[l]] / denom_l

                        I00n = I00_arr[n]
                        I10n = I_ab(1, 0, Q_arr[n], n_arr[n])
                        I01n = I_ab(0, 1, Q_arr[n], n_arr[n])
                        chi1l_n = chi1_arr[n][l]
                        brack_l_n = (I00n * (a_arr[n][l] + c_arr[n][l])
                                     + I10n * chi1l_n
                                     + I01n * np.conj(chi1l_n))

                        # Product of I00 over modes other than m and n
                        if m == n:
                            prod_other = np.prod([I00_arr[mm] for mm in range(Nm)
                                                  if mm != m])
                        else:
                            prod_other = np.prod([I00_arr[mm] for mm in range(Nm)
                                                  if mm != m and mm != n])

                        R2 += C_mn_kl * F * brack_k_m * brack_l_n * prod_other

        # Second sum in R2 (same-mode cross terms): k<l, single mode m
        for k in range(1, K + 2):
            denom_k = mu0[e_full[k-1], e_full[k]]
            for l in range(k + 1, K + 2):
                denom_l = mu0[e_full[l-1], e_full[l]]
                if denom_k == 0.0 or denom_l == 0.0:
                    continue
                for m in range(Nm):
                    C_mk = C * mu1_list[m][e_full[k-1], e_full[k]] / denom_k
                    C_nl = C * mu1_list[m][e_full[l-1], e_full[l]] / denom_l

                    I00m = I00_arr[m]
                    I10m = I_ab(1, 0, Q_arr[m], n_arr[m])
                    I01m = I_ab(0, 1, Q_arr[m], n_arr[m])
                    I20m = I_ab(2, 0, Q_arr[m], n_arr[m])
                    I02m = I_ab(0, 2, Q_arr[m], n_arr[m])
                    J11m = J_11(Q_arr[m], n_arr[m])

                    chi1k = chi1_arr[m][k]
                    chi1l = chi1_arr[m][l]
                    chi_kl = chi(m, k, l-1, taus, omega_vib)   # chi^(m)_{k,l-1}

                    prod_other = np.prod([I00_arr[mm] for mm in range(Nm) if mm != m])

                    # The bracket from Eq. (9) second sum
                    term2 = (I20m * chi1k * chi1l
                             + I02m * np.conj(chi1k) * np.conj(chi1l)
                             + J11m * (chi1k * np.conj(chi1l) + chi1l * np.conj(chi1k))
                             + I00m * chi_kl
                             - (I10m * chi1k + I01m * np.conj(chi1k))
                               * (I10m * chi1l + I01m * np.conj(chi1l)))

                    R2 += C_mk * C_nl / C * F * term2 * prod_other

    # ── R3: linear in mu2 (Eq. 10) ────────────────────────────────────────
    R3 = 0.0 + 0.0j
    if use_HT3 and mu2_list is not None:
        for k in range(1, K + 2):
            denom_k = mu0[e_full[k-1], e_full[k]]
            if denom_k == 0.0:
                continue
            for m in range(Nm):
                # C^(m)_{|k} = C * mu2[m][e_{k-1},e_k] / mu0[e_{k-1},e_k]
                C_mk2 = C * mu2_list[m][e_full[k-1], e_full[k]] / denom_k

                I00m = I00_arr[m]
                I10m = I_ab(1, 0, Q_arr[m], n_arr[m])
                I01m = I_ab(0, 1, Q_arr[m], n_arr[m])
                I20m = I_ab(2, 0, Q_arr[m], n_arr[m])
                I02m = I_ab(0, 2, Q_arr[m], n_arr[m])
                I11m = I_ab(1, 1, Q_arr[m], n_arr[m])
                ac_k = a_arr[m][k] + c_arr[m][k]
                chi1k = chi1_arr[m][k]

                prod_other = np.prod([I00_arr[mm] for mm in range(Nm) if mm != m])

                bracket = (I00m
                           + I00m * ac_k**2
                           + I20m * chi1k**2
                           + I02m * chi1k * np.conj(chi1k)
                           + 2.0 * I11m
                           + (I10m + I01m) * ac_k)

                R3 += C_mk2 * F * bracket * prod_other

    return R0 + R1 + R2 + R3


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK 3 — Numerical evaluation, output files, contour plots
# ══════════════════════════════════════════════════════════════════════════════

def tau_values(tau_coeffs, t_vec):
    """
    Evaluate the τ_k values given a dict of t-values.
    tau_coeffs: list of dicts {t_index: coeff}
    t_vec: 1-D array or list of length M (t_vec[i] = t_{i+1})
    Returns array of shape (K,)
    """
    taus = []
    for coeffs in tau_coeffs:
        val = sum(c * t_vec[l - 1] for l, c in coeffs.items())
        taus.append(val)
    return np.array(taus)


def compute_fm(m_idx, taus, e_seq, z, omega_vib):
    """
    Compute f_m for mode m_idx given the tau values (Eq. 4).
    e_seq : list of ints, length K  (e_1..e_K, e_0=e_{K+1}=0 added below)
    z     : (N_elec, Nm) displacement array
    Returns complex scalar.
    """
    # Add e_0 = 0 and e_{K+1} = 0
    e_full = [0] + list(e_seq) + [0]   # length K+2
    K = len(e_seq)
    om = omega_vib[m_idx] / HBAR_meV_fs  # convert meV → rad/fs

    # z^(m)_{j,l} = z[j][m] - z[l][m]
    def zdiff(j, l):
        return z[j, m_idx] - z[l, m_idx]

    fm = 0.0 + 0.0j
    for k in range(1, K + 1):
        for l in range(1, K - k + 2):
            z1 = zdiff(e_full[l - 1], e_full[l])
            z2 = zdiff(e_full[l + k - 1], e_full[l + k])
            # Product exp(-i*om*tau_p) for p = l..l+k-1  (1-based tau index)
            prod = 1.0 + 0.0j
            for p in range(l, l + k):
                prod *= np.exp(-1j * om * taus[p - 1])
            fm += z1 * z2 * (1.0 - prod)
    return fm


def compute_response(t_free, free_axes, params, e_seq, tau_coeffs):
    """
    Compute R^(e,M) * R^(v,M) on a 2D grid defined by t_free.

    t_free    : 1-D array of time values (fs) for both free axes
    free_axes : list of two 1-based time indices that are free
    params    : parameter dict
    Returns complex 2D array of shape (N_t, N_t).
    """
    M          = params["M"]
    epsilon    = np.array(params["epsilon"])
    tau_dep    = params["tau_dep"]   # fs
    tau_rel    = params["tau_rel"]   # fs
    mu         = params["mu_0"]
    omega_vib  = np.array(params["omega_vib"])
    z          = params["z"]
    T          = params["T"]
    fixed      = params["fixed_times"]
    kets       = params["kets"]
    bras       = params["bras"]
    x_val      = compute_x(params)
    N_t        = len(t_free)
    Nm         = len(omega_vib)
    K          = len(e_seq)

    # c_k coefficients: per time-step damping rates (units: 1/fs)
    #   c_k = 1/tau_dep  if bra_k != ket_k          (dephasing)
    #   c_k = 1/tau_rel  if bra_k == ket_k != 0     (relaxation)
    #   c_k = 0          if bra_k == ket_k == 0     (ground state)
    c_vec = np.zeros(M + 1)
    for k in range(1, M + 1):
        if bras[k] != kets[k]:
            c_vec[k] = 1.0 / tau_dep
        elif kets[k] != 0:
            c_vec[k] = 1.0 / tau_rel
        # else c_k = 0

    # C is always carried inside compute_R_vib_full (Eq. 11 / Eqs. 7-10).
    # R_elec = (-1)^x * exp[phase] * exp[decay]  — no C here.
    prefactor = ((-1) ** x_val)

    R = np.zeros((N_t, N_t), dtype=complex)

    axis1, axis2 = free_axes  # 1-based

    for i, ta in enumerate(t_free):
        for j, tb in enumerate(t_free):
            # Assemble t_vec of length M  (t_vec[k-1] = t_k)
            t_vec = []
            for idx in range(1, M + 1):
                if idx == axis1:
                    t_vec.append(ta)
                elif idx == axis2:
                    t_vec.append(tb)
                elif idx in fixed:
                    t_vec.append(fixed[idx])
                else:
                    t_vec.append(0.0)

            # Compute τ_k (for vibrational component)
            taus = tau_values(tau_coeffs, t_vec)

            # Electronic factor (Eq. 11):
            # (-1)^x * exp[-i*sum_k t_k*(eps_{ket_k}-eps_{bra_k})/hbar] * exp[-sum_k c_k*t_k]
            # C is in R_elec (non-HT path) or in R_vib via compute_R_vib_full (HT path)
            phase_osc  = sum(t_vec[k-1] * (epsilon[kets[k]] - epsilon[bras[k]])
                             / HBAR_meV_fs
                             for k in range(1, M + 1))
            decay      = sum(c_vec[k] * t_vec[k-1] for k in range(1, M + 1))
            R_elec = prefactor * np.exp(-1j * phase_osc) * np.exp(-decay)

            # Vibrational component: always via compute_R_vib_full
            # (C is inside; HT flags off = pure Franck-Condon)
            R_vib = compute_R_vib_full(
                e_seq, taus, z, omega_vib, T,
                mu, params.get('mu1'), params.get('mu2'),
                params.get('use_HT1', False),
                params.get('use_HT2', False),
                params.get('use_HT3', False))

            R[i, j] = R_elec * R_vib

    return R


def fft2d(R, t_free):
    """
    Compute the 2D inverse FT of R(t1, t2) → R̃(ω1, ω2):

      R̃(ω1,ω2) = Σ_{n1,n2} R(n1·Δt, n2·Δt) · exp(+i·ω1·n1·Δt) · exp(+i·ω2·n2·Δt)

    Implemented as N²·ifft2 (ifft2 includes a 1/N² normalisation that we undo).
    Returns (omega_axis_meV, R_freq) where ℏω is in meV.
    """
    N   = len(t_free)
    dt  = t_free[1] - t_free[0]  # fs

    # ifft2 = (1/N²) Σ R · exp(+i·2π·n·k/N); multiply by N² to get unnormalised sum
    R_fft = fftshift(np.fft.ifft2(R)) * N**2
    # Frequency axis: ℏω_k = ℏ · 2π·k/(N·Δt), in meV
    freq  = fftshift(fftfreq(N, d=dt))          # cycles/fs
    omega = freq * 2 * np.pi * HBAR_meV_fs      # ℏω in meV
    return omega, R_fft


def write_output(filename, axis1, axis2, R, axis_label1, axis_label2):
    """Write a 5-column output file."""
    N = R.shape[0]
    with open(filename, "w") as f:
        f.write(f"# {axis_label1}   {axis_label2}   Re(R)   Im(R)   |R|\n")
        for i in range(N):
            for j in range(N):
                f.write(f"{axis1[i]:.6e}  {axis2[j]:.6e}  "
                        f"{R[i, j].real:.6e}  {R[i, j].imag:.6e}  "
                        f"{abs(R[i, j]):.6e}\n")
    print(f"  [Block 3] Output written to: {filename}")


def _plot3D2(fig, X, Y, Z, s1, s2, xlabel, ylabel, subplot_position):
    """
    Add one subplot panel using pcolormesh + overlaid gray contours with labels.
    Matches the style of the user's plot3D2 function:
      - pcolormesh with RdBu and gouraud shading as background
      - contour with gray colormap overlaid, with inline labels
      - colorbar on each panel
      - text labels s1 (bottom-left) and s2 (bottom-right) in data coordinates
    """
    ax = fig.add_subplot(subplot_position)

    # pcolormesh background
    vmax = np.max(np.abs(Z))
    if vmax == 0:
        vmax = 1.0
    plot = ax.pcolormesh(X, Y, Z, cmap='RdBu', shading='gouraud',
                         vmin=-vmax, vmax=vmax)
    fig.colorbar(plot, ax=ax)

    # Overlaid gray contours with inline labels
    cset = ax.contour(X, Y, Z, cmap='gray')
    ax.clabel(cset, inline=True)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    # Text labels: placed at 5% and 75% of x-range, 5% from bottom of y-range
    xlo, xhi = X.min(), X.max()
    ylo       = Y.min()
    yspan     = Y.max() - ylo
    ax.text(xlo + 0.05 * (xhi - xlo), ylo + 0.05 * yspan, s1, fontsize=15)
    ax.text(xlo + 0.75 * (xhi - xlo), ylo + 0.05 * yspan, s2, fontsize=15)

    return ax


def make_contour_plots(axis1, axis2, R, xlabel, ylabel, title_prefix, outfile_prefix,
                       axis_range=None):
    """
    Produce three panels (real, imaginary, absolute value) using the plot3D2 style:
    pcolormesh (RdBu, gouraud) + overlaid gray contours with inline labels.
    axis_range: optional (min, max) tuple to restrict both axes (in axis units).
    """
    components = [
        (R.real,    "Real part",      "Re"),
        (R.imag,    "Imaginary part", "Im"),
        (np.abs(R), "Absolute value", "|R|"),
    ]

    # Apply axis_range mask to avoid passing huge arrays to pcolormesh
    ax1, ax2 = axis1, axis2
    data_R = R
    if axis_range is not None:
        lo, hi = axis_range
        mask1 = (ax1 >= lo) & (ax1 <= hi)
        mask2 = (ax2 >= lo) & (ax2 <= hi)
        ax1 = ax1[mask1]
        ax2 = ax2[mask2]
        data_R = R[np.ix_(mask1, mask2)]

    A1, A2 = np.meshgrid(ax2, ax1)   # rows=axis1, cols=axis2

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(title_prefix, fontsize=13)

    for idx, (full_data, label, tag) in enumerate(components):
        # Apply same mask
        if axis_range is not None:
            data = full_data[np.ix_(mask1, mask2)]
        else:
            data = full_data

        s1 = label
        s2 = ""
        _plot3D2(fig, A2, A1, data, s1, s2, xlabel, ylabel, 131 + idx)

    plt.tight_layout()
    fname = f"{outfile_prefix}_contour.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Block 3] Contour plot saved to: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python response_functions.py input.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    base = os.path.splitext(input_file)[0]

    # ── Block 1 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BLOCK 1 — Reading input and identifying response function")
    print("=" * 70)

    params = parse_input(input_file)
    M      = params["M"]
    kets   = params["kets"]
    bras   = params["bras"]

    print(f"\n  Order of light-matter interaction:  M = {M}")
    print(f"  Ket sequence:  " + "  ".join(f"|{k}⟩" for k in kets))
    print(f"  Bra sequence:  " + "  ".join(f"⟨{b}|" for b in bras))
    print(f"  Temperature:      T       = {params['T']} K")
    print(f"  Dephasing time:   τ_dep   = {params['tau_dep']} fs")
    print(f"  Relaxation time:  τ_rel   = {params['tau_rel']} fs")
    print(f"  Electronic energies (meV): {params['epsilon']}")
    print(f"  Dipole matrix μ₀:\n{params['mu_0']}")
    print(f"  HT flags: use_HT1={params['use_HT1']}, "
          f"use_HT2={params['use_HT2']}, use_HT3={params['use_HT3']}")
    print(f"  Fixed times: {params['fixed_times']}")

    # Determine free axes (all t_1..t_M not in fixed_times)
    fixed_keys = set(params["fixed_times"].keys())
    free_axes  = [i for i in range(1, M + 1) if i not in fixed_keys]
    if len(free_axes) != 2:
        raise ValueError(f"Expected exactly 2 free time axes, got {len(free_axes)}: {free_axes}. "
                         f"Please fix (M-2)={M-2} times in fixed_times.")
    print(f"  Free time axes: t_{free_axes[0]}, t_{free_axes[1]}")

    e_seq, tau_coeffs = build_ek_tau(params)

    # Draw Feynman diagram
    feynman_file = f"{base}_feynman.png"
    draw_feynman_diagram(params, e_seq, tau_coeffs, outfile=feynman_file)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    tex_file, pdf_file = print_response_function_expression(
        params, e_seq, tau_coeffs, outfile_prefix=base)

    # ── Block 3 ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  BLOCK 3 — Numerical evaluation, output files, and plots")
    print("=" * 70)

    N_t   = params["N_t"]
    t_max = params["t_max"]
    t_arr = np.linspace(0, t_max, N_t, endpoint=False)

    print(f"\n  Computing R(t_{free_axes[0]}, t_{free_axes[1]}) on a {N_t}×{N_t} grid "
          f"[0, {t_max} fs] ...")

    R_time = compute_response(t_arr, free_axes, params, e_seq, tau_coeffs)

    # Electronic and vibrational components separately
    # (recompute by setting vibrational / electronic factor = 1)
    # For output we split: R_elec and R_vib (product gives R_total)
    # We write ONE time-domain file for the total response
    time_file = f"{base}_time_domain.txt"
    write_output(time_file, t_arr, t_arr, R_time,
                 f"t_{free_axes[0]} (fs)", f"t_{free_axes[1]} (fs)")

    # Contour plots — time domain
    make_contour_plots(t_arr, t_arr, R_time,
                       xlabel=f"$t_{{{free_axes[0]}}}$ (fs)",
                       ylabel=f"$t_{{{free_axes[1]}}}$ (fs)",
                       title_prefix=f"Time-domain response function  R$^{{({M})}}$  "
                                    f"(T={params['T']} K)",
                       outfile_prefix=f"{base}_time")

    # Frequency domain via 2D FFT
    print(f"\n  Computing 2D FFT → frequency domain ...")
    omega, R_freq = fft2d(R_time, t_arr)

    freq_file = f"{base}_freq_domain.txt"
    write_output(freq_file, omega, omega, R_freq,
                 f"hbar*omega_{free_axes[0]} (meV)", f"hbar*omega_{free_axes[1]} (meV)")

    freq_range = params.get("freq_range", None)
    omega_min, omega_max = omega.min(), omega.max()
    if freq_range is not None:
        req_min, req_max = freq_range
        clipped_min = max(req_min, omega_min)
        clipped_max = min(req_max, omega_max)
        if clipped_min != req_min or clipped_max != req_max:
            print(f"  [Block 3] Note: requested freq_range "
                  f"[{req_min:.4g}, {req_max:.4g}] meV exceeds the FFT range "
                  f"[{omega_min:.4g}, {omega_max:.4g}] meV. "
                  f"Using [{clipped_min:.4g}, {clipped_max:.4g}] meV instead.")
        freq_range = (clipped_min, clipped_max)
    make_contour_plots(omega, omega, R_freq,
                       xlabel=f"$\\hbar\\omega_{{{free_axes[0]}}}$ (meV)",
                       ylabel=f"$\\hbar\\omega_{{{free_axes[1]}}}$ (meV)",
                       title_prefix=f"Frequency-domain response function  R$^{{({M})}}$  "
                                    f"(T={params['T']} K)",
                       outfile_prefix=f"{base}_freq",
                       axis_range=freq_range)

    print("\n  ✓ Block 3 complete.")
    print("\n  Output files:")
    print(f"    {feynman_file}")
    print(f"    {tex_file}")
    if pdf_file:
        print(f"    {pdf_file}")
    print(f"    {time_file}")
    print(f"    {freq_file}")
    print(f"    {base}_time_contour.png")
    print(f"    {base}_freq_contour.png")



if __name__ == "__main__":
    main()


