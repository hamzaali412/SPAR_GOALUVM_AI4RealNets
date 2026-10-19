# SPAR: Splitting and Power dispatch with Adversarial Risk-awareness

## Note on the Sim2Real principle

The code uses the original Sim2Real environment as provided by the organizers, without modifying it.

To build its AC power-flow model, the code reads static network parameters (line resistance,
reactance, line charging and transformer taps) from the environment provided.
In the Sim2Real setting of the competition, these belong to the real environment, which the
organizers intended to be accessible only through observations. In a real Sim2Real deployment, these
parameters are to be estimated from measurements.

To follow the Sim2Real principle, take the parameters from the simulation model attached to each
observation instead: in `submission/ACOPF/ACOPF_mod/obs_network.py`, function `model_grid`, use the
commented line `grid = obs._obs_env.backend._grid`.

## Installation

Python 3.10:

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

Set the variables at the top of `main.py`:

- `AGENT`: `"trad"` or `"do_nothing"`
- `ENV`: grid2op environment name (`l2rpn_idf_2023`, downloaded on first use) or a local path to it
- `SCENARIO`: week to play
- `MAX_STEPS`: `None` for the full week (2016 steps), or a smaller number

Then run:

```bash
python main.py
```

## License

MIT (see `LICENSE`). The bundled IPOPT binaries are distributed under their own licenses, listed in
`submission/ipopt_testing/THIRD_PARTY_NOTICES.md`.

## About

SPAR is the approach of the University of Vermont's Grid Optimization and Analysis Lab (GOAL) for the
L2RPN 2026 competition (AI4REALNET).

The team includes Hamza Ali, Amritanshu Pandey and Samuel Chevalier.
