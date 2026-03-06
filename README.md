# Response Function Calculator for Coherent Multidimensional Spectroscopy

A Python tool for computing and visualising nonlinear optical response functions in the **multimode displaced harmonic oscillator (DHO) model**, with support for **Herzberg–Teller (HT) vibronic coupling** corrections.

Given a Feynman diagram specified by its ket/bra state sequences, the code automatically derives the response function, produces a symbolic LaTeX document with all equations written out explicitly for the chosen diagram, and evaluates the function numerically on a 2D time grid before Fourier transforming to the frequency domain.

---

## Features

- **Arbitrary order** *M* and diagram topology: specify any ket/bra sequence consistent with the DHO model.
- **Franck–Condon (FC) vibrational response** via the exact multimode DHO expression, valid at arbitrary temperature *T*.
- **Herzberg–Teller corrections**:
  - *R*₁: linear in the non-Condon dipole operator μ̂₁
  - *R*₂: quadratic in μ̂₁
  - *R*₃: linear in the second-order non-Condon operator μ̂₂
- **Symbolic LaTeX output**: all relevant equations are written out with diagram-specific indices substituted and fixed waiting times replaced by their numerical values, leaving only physical parameters symbolic. Compiles to PDF via `pdflatex`.
- **Feynman diagram**: automatically drawn and saved as a PNG.
- **2D time-domain and frequency-domain output**: numerical grids written to text files; contour plots with overlaid labeled contours produced for real part, imaginary part, and absolute value.
- **2D inverse Fourier transform**: ℝ̃(ω₁,ω₂) = Σ R(n₁Δt, n₂Δt) exp(+iω₁n₁Δt) exp(+iω₂n₂Δt).
- **T = 0 support**: ⟨n_m⟩ = 0 is handled exactly (I₀₀ = 1, all other I_ab = 0).

---

## Installation

### Requirements

- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib
- `pdflatex` (optional, for PDF compilation of the equation sheet)

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

On most Linux systems `pdflatex` is available via `texlive-latex-base`. On macOS it comes with MacTeX. Set `compile_latex = false` in the input file to skip PDF generation entirely.

---

## Usage

```bash
python response_functions.py input.txt
```

All output files are written to the same directory as the input file, with the input filename (minus extension) as prefix. For an input file named `mysample.txt` the outputs are:

| File | Contents |
|------|----------|
| `mysample_feynman.png` | Feynman diagram |
| `mysample_equations.tex` | LaTeX source for the equation sheet |
| `mysample_equations.pdf` | Compiled equation sheet (if `compile_latex = true`) |
| `mysample_time_domain.txt` | R(t₁, t₂) on the 2D grid (5 columns: t₁, t₂, Re, Im, \|R\|) |
| `mysample_freq_domain.txt` | R̃(ω₁, ω₂) on the 2D grid (5 columns: ℏω₁, ℏω₂, Re, Im, \|R̃\|) |
| `mysample_time_contour.png` | Contour plots of R (real, imaginary, absolute value) |
| `mysample_freq_contour.png` | Contour plots of R̃ |

---

## Input file format

Lines beginning with `#` are comments. All times are in **femtoseconds**, all energies in **meV**.

### Mandatory parameters

| Key | Description |
|-----|-------------|
| `M` | Order of the light–matter interaction (number of field interactions) |
| `kets` | M+1 integers: electronic state of the ket at each time step (0-indexed; starts and ends at 0) |
| `bras` | M+1 integers: electronic state of the bra at each time step |
| `epsilon` | Electronic energies in meV (*N*_elec values; *N*_elec is inferred as max state index + 1) |
| `omega_vib` | Vibrational mode frequencies in meV (*N*_m values) |
| `z` | Dimensionless displacements: *N*_elec rows × *N*_m columns |
| `mu_0` | Permanent transition dipole matrix: *N*_elec × *N*_elec real symmetric matrix |
| `fixed_times` | Which M−2 time indices (1-based) are fixed and at what values, e.g. `2:100.0  3:0.0` |

### Optional parameters

| Key | Default | Description |
|-----|---------|-------------|
| `T` | 300 | Temperature in K |
| `tau_dep` | 100000 | Dephasing time in fs |
| `tau_rel` | 100000 | Relaxation time in fs |
| `t_max` | 500 | Upper limit of the time grid in fs |
| `N_t` | 256 | Number of time points per axis |
| `freq_range` | full FFT range | Plot range for frequency-domain spectra in meV, format `min:max` |
| `compile_latex` | true | Whether to compile the `.tex` file to PDF |
| `use_HT1` | false | Include R₁ (linear in μ̂₁) |
| `use_HT2` | false | Include R₂ (quadratic in μ̂₁) |
| `use_HT3` | false | Include R₃ (linear in μ̂₂) |
| `mu1_1` … `mu1_Nm` | — | Non-Condon μ̂₁ matrices (required if `use_HT1` or `use_HT2` is true) |
| `mu2_1` … `mu2_Nm` | — | Second-order μ̂₂ matrices (required if `use_HT3` is true) |

### Constraints on kets/bras

- At each time step *i* = 1…*M*, exactly one of {ket, bra} may change state; simultaneous changes on both sides are not allowed.
- The sequences must start and end at state 0 (ground state).

### Frequency grid and the Nyquist condition

The time step is Δt = `t_max` / `N_t`. The frequency axis covers ℏω ∈ [−ℏπ/Δt, +ℏπ/Δt] with resolution δ(ℏω) = 2πℏ / `t_max`. To resolve features at energy *E* (meV) the Nyquist condition requires:

```
t_max  <  π * ℏ / E  ×  N_t       (coverage)
t_max  >  2π * ℏ / δE              (resolution)
```

where ℏ = 0.6582 meV·fs.

---

## Example

The file `example_input.txt` demonstrates an *M* = 4 rephasing-type diagram with 2 electronic states and 2 vibrational modes. Run it with:

```bash
python response_functions.py example_input.txt
```

To activate Herzberg–Teller corrections, set `use_HT1 = true` (and/or `use_HT2`, `use_HT3`) and provide the corresponding `mu1_m` and/or `mu2_m` matrices.

---

## Theory

The response function factorises as

$$R^{(M)}_T = R^{(v,M)}_T \times (-1)^x \exp\!\left[-\frac{i}{\hbar}\sum_k t_k(\varepsilon_{\mathrm{ket}_k}-\varepsilon_{\mathrm{bra}_k})\right] \exp\!\left[-\sum_k c_k t_k\right]$$

where *x* counts bra-side arrows mod 2, and *c_k* = 1/τ_dep (dephasing), 1/τ_rel (relaxation), or 0 (ground state).

The vibrational factor R^(v,M)_T is computed in the displaced harmonic oscillator model following the equations of the accompanying PDF equation sheet, with optional Herzberg–Teller corrections R₁, R₂, R₃ as described in the reference below.

The χ functions entering the vibrational expressions are defined as

$$\chi^{(m)}_{k,l} = e^{-i\omega_m(t_k + \cdots + t_l)}, \qquad \chi^{(m)}_{k,k-1} = 1$$

### Physical units

| Quantity | Unit |
|----------|------|
| Time | fs |
| Energy / frequency | meV (via ℏ = 0.6582119569 meV·fs) |
| Temperature | K (*k*_B = 0.08617333 meV/K) |
| Displacements *z* | dimensionless |

---

## Repository structure

```
.
├── response_functions.py   # Main script (all blocks)
├── example_input.txt       # Example input (M=4, 2 states, 2 modes)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

If you use this code in published work, please cite:

*(placeholder — to be updated)*
