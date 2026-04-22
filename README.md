# Neural Network Bootstrap for CFT Correlation Functions

This repository contains standalone Python scripts demonstrating a neural
network approach to the reconstruction of conformal correlators as
described in
> **Neural Networks Reveal a Universal Bias in Conformal Correlators**  
> Kausik Ghosh, Sidhaarth Kumar, Vasilis Niarchos, and Andreas Stergiou (2026)  
> [arXiv:2604.18673](https://arxiv.org/abs/2604.18673)

> **Neural Spectral Bias and Conformal Correlators I: Introduction and Applications**  
> Kausik Ghosh, Sidhaarth Kumar, Vasilis Niarchos, and Andreas Stergiou (2026)  
> [arXiv:2604.18686](https://arxiv.org/abs/2604.18686)

Each script trains a small ensemble of networks to learn the four-point
function G(z) of a scalar operator in a specific CFT by enforcing crossing
symmetry as the only dynamical constraint, supplemented by one or
more anchor conditions that fix the value of G at known reference points.

## Method

A neural network parametrises G(z), with a physics-motivated pre-factor
that encodes the expected singular behaviour near z = 0 and z = 1. The
network is trained to satisfy the crossing equation

```
G(z) = (z / (1-z))^(2*Delta) * G(1-z)
```

via a normalised residual loss, together with an anchor loss that penalises
deviations from the exact (or bootstrap) value at one or more fixed points.
Running multiple independent training runs with different random seeds
produces an ensemble whose spread quantifies the remaining ambiguity in the
solution.

## Scripts

| Script | System | Reference |
|--------|--------|-----------|
| `run_2d_minimal_models.py` | 2D minimal models M(m, m+1)        | Exact hypergeometric formula |
| `run_gfb.py`               | Generalised free boson             | Exact closed-form formula    |
| `run_3d_ising_sigma.py`    | 3D Ising sigma operator            | Conformal bootstrap sum      |
| `run_3d_ising_epsilon.py`  | 3D Ising epsilon operator          | Conformal bootstrap sum      |
| `run_ads2_contact.py`      | AdS2 contact diagram (phi^4, Delta_phi=1) | Exact closed-form formula    |

## Installation

```bash
pip install torch numpy scipy mpmath matplotlib
```

The 3D Ising scripts additionally require `mpmath` for arbitrary-precision
hypergeometric functions used in the conformal block evaluation.

## Usage

Each script accepts a `--num-runs` flag controlling how many independent
training runs to perform. Start with a small number (5–10) on a laptop.

### 2D Minimal Models

```bash
# Ising model (m=3, default)
python run_2d_minimal_models.py --num-runs 10

# Tricritical Ising (m=4)
python run_2d_minimal_models.py --num-runs 10 --m 4 --output-dir results_mm4

# Multiple anchor points
python run_2d_minimal_models.py --num-runs 10 --anchor-points 0.2 0.5 0.8
```

The `--m` parameter sets the minimal model index: m=3 is the Ising model,
m=4 is the tricritical Ising model, etc. Fractional values (e.g. `--m 7/2`)
are also accepted.

### Generalised Free Boson

```bash
# Default (Delta = 0.5)
python run_gfb.py --num-runs 10

# Different conformal dimension
python run_gfb.py --num-runs 10 --delta 0.25 --output-dir results_gfb025
python run_gfb.py --num-runs 10 --delta 1/4   # fractions accepted
```

### 3D Ising Sigma

```bash
python run_3d_ising_sigma.py --num-runs 5
python run_3d_ising_sigma.py --num-runs 5 --anchor-points 0.2 0.4
```

The reference answer is a truncated sum of diagonal conformal blocks using
OPE data from the conformal bootstrap. Computing the conformal blocks via
`mpmath` is slow; expect a few extra minutes of setup per run on a laptop.

### 3D Ising Epsilon

```bash
python run_3d_ising_epsilon.py --num-runs 5
python run_3d_ising_epsilon.py --num-runs 5 --anchor-points 0.2 0.4
```

Same setup and caveats as the sigma script, with the epsilon OPE data.

### AdS2 Contact Diagram

```bash
python run_ads2_contact.py --num-runs 10
python run_ads2_contact.py --num-runs 10 --output-dir results_ads2
python run_ads2_contact.py --num-runs 5 --anchor-points 0.3 0.5
```

Learns the four-point function G(z) of a scalar with Delta_phi=1 in AdS2
from a phi^4 contact interaction. The function is split as G(z) = H(z) + L(z),
where L(z) = 2 z^2 (log(z) - 1) is a known leading contribution and H(z) is learned by the
network. The exact answer is known analytically.

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-runs N` | 10 (5 for 3D) | Independent training runs |
| `--output-dir PATH` | script-specific | Directory for results and plots |
| `--anchor-points z1 z2 ...` | `0.3` (0.4 for AdS2) | z values where exact/bootstrap value is enforced |
| `--max-epochs N` | 200000 | Maximum training epochs per run |
| `--patience N` | 5000 | Early-stopping patience (epochs without improvement) |
| `--z-min`, `--z-max` | 0.01, 0.9 | Training and evaluation domain |
| `--n-points N` | 1000 (600 for 3D) | Points in the z grid |
| `--save-models` | off | Save model weights as `.pt` files |

## Outputs

Each script writes results to the chosen output directory:

```
results_*/
├── metadata.json          # Configuration and reference values
├── model_000.json         # Per-run results (loss, epochs, predictions)
├── model_001.json
├── ...
└── ensemble_results.png   # Two-panel figure: ensemble vs reference + error
```

The figure shows all individual runs as thin lines, the ensemble mean with
a ±1σ band, the exact or bootstrap reference, and the relative error of the
mean normalised by |G(z)| + 1.


## Citation

If you use this code, please cite the papers

```bibtex
@article{Ghosh:2026jbw,
    author = "Ghosh, Kausik and Kumar, Sidhaarth and Niarchos, Vasilis and Stergiou, Andreas",
    title = "{Neural Networks Reveal a Universal Bias in Conformal Correlators}",
    eprint = "2604.18673",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "ITCP-2026-4, CCTP-2026-4",
    month = "4",
    year = "2026"
}

@article{Ghosh:2026xnp,
    author = "Ghosh, Kausik and Kumar, Sidhaarth and Niarchos, Vasilis and Stergiou, Andreas",
    title = "{Neural Spectral Bias and Conformal Correlators I: Introduction and Applications}",
    eprint = "2604.18686",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "ITCP-2026-5, CCTP-2-26-5",
    month = "4",
    year = "2026"
}
```

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
