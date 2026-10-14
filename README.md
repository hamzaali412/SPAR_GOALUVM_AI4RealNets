# TRAD: Topology and Risk-Aware AC Dispatch

## Environment

| | |
|---|---|
| OS | Linux x86-64 (other systems: install IPOPT yourself, see below) |
| Python | 3.10 |
| Packages | grid2op 1.12.2, lightsim2grid 0.13.1, pyomo 6.10.1, numpy 1.26.4, scipy 1.15.3, networkx 3.4.2 |
| NLP solver | IPOPT 3.14 (bundled in `submission/ipopt_testing`) |
| Grid2Op environment | `l2rpn_idf_2023`, `competition` difficulty (downloaded automatically on first use) |

## Installation

```bash
conda create -n trad python=3.10
conda activate trad
pip install -r requirements.txt
```

IPOPT binaries for Linux x86-64 are bundled and used automatically. On other systems, or to use your
own build:

```bash
conda install -c conda-forge ipopt
```

## Usage

Edit the settings at the top of `main.py`:

```python
AGENT = "trad"               # "trad" or "do_nothing"
ENV = "l2rpn_idf_2023"       # grid2op environment name, or a local path to it
SCENARIO = 0                 # week to play
MAX_STEPS = None             # None = full week (2016 steps)
```

Then run:

```bash
python main.py
```

## Repository layout

```
main.py              run one scenario
requirements.txt
submission/
  make_agent.py      agent entry point
  ACOPF/ACOPF_mod/   controller code
  ipopt_testing/     bundled IPOPT binaries and licenses
```

## Sources 

- IPOPT: A. Wächter, L. T. Biegler, *Mathematical Programming* 106(1), 25–57, 2006 (EPL-2.0).
  Bundled binaries and their libraries are credited in `submission/ipopt_testing/THIRD_PARTY_NOTICES.md`.
- Busbar splitting: A. Marot, B. Donnot, S. Tazi, P. Panciatici, "Expert system for topological
  remedial action discovery in smart grids," MedPower 2018.
- [Grid2Op](https://github.com/Grid2op/grid2op) and [LightSim2Grid](https://github.com/Grid2op/lightsim2grid).
