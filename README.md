# Sweet Gradient

![Tests](https://github.com/asouchaud/Sweet_Gradient/actions/workflows/tests.yml/badge.svg?branch=main)

Code to reproduce the DAFx 2026 figures. Each figure script runs the optimisation and writes plots under `results/`.

## Installation

Python 3.12 is required.

```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt
```

## Generate the figures

```bash
python DAFx2026_fig_rendering.py
python plot_fig5_from_results.py
python DAFx2026_fig_VBAP.py
python DAFx2026_fig_CTC.py
python DAFx2026_fig_beamforming.py
```

`plot_fig5_from_results.py` rebuilds the left- and right-ear figure 5 PDFs from the four most recent rendering runs.

## Tests

```bash
python run_tests.py
```

## Layout

- `DAFx2026_fig_*.py` are the figure entry points
- `scripts/` contains propagation, perceptual models, optimisation, plotting, and saving
- `data/` contains the HRIRs, gammatone filter bank, and comparison curves used by those scripts
- `tests/` contains unit and integration tests
