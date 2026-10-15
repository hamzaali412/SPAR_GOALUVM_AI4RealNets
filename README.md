# TRAD: Topology and Risk-Aware AC Dispatch

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
