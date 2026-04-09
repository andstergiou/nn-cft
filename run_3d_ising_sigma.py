#!/usr/bin/env python3
"""
Neural network bootstrap for the 3D Ising sigma four-point function.

Trains an ensemble of neural networks to learn the four-point function G(z)
of the sigma operator (Delta_sigma = 0.51815) in the 3D Ising CFT by
enforcing the crossing symmetry constraint

    G(z) = (z / (1-z))^(2*Delta_sigma) * G(1-z).

The reference ("bootstrap") answer is computed as a sum of diagonal conformal
blocks using operator dimensions and OPE coefficients from the conformal
bootstrap literature (Simmons-Duffin 2016). This bootstrap sum is used only
to set the anchor constraint and to assess accuracy.

NOTE: Computing the bootstrap reference involves mpmath hypergeometric
functions and is the most expensive part of setup. With the default anchor
point [0.3], this is evaluated just once per run. Expect each training run to
take several minutes on a laptop.

Usage (personal computer):
    python run_3d_ising_sigma.py --num-runs 5
    python run_3d_ising_sigma.py --num-runs 10 --output-dir results_sigma
    python run_3d_ising_sigma.py --num-runs 3 --anchor-points 0.2 0.4

Dependencies: torch, numpy, scipy, mpmath, matplotlib
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpmath import hyper
from scipy.special import comb, poch


device = torch.device("cpu")
EPS = 1e-8
ANCHOR_WEIGHT = 100.0

# Default conformal dimension for the 3D Ising sigma operator
DELTA_SIGMA = 0.51815

# ---------------------------------------------------------------------------
# Bootstrap data: (spin n, dimension Delta, OPE coefficient squared a)
# Operators appearing in the sigma x sigma OPE of the 3D Ising CFT.
# Data from Simmons-Duffin (2016), arXiv:1612.08471.
# ---------------------------------------------------------------------------
DATA_BOOTSTRAP = [
    (0,      1.41263,   1.1064),
    (0,      3.82968,   0.00281027),
    (0,      6.8956,    5.38462e-7),
    (0,      7.2535,    2.6244e-8),
    (2 // 2, 3.0,       0.283642),
    (2 // 2, 5.50915,   0.000298187),
    (4 // 2, 5.02267,   0.01745),
    (4 // 2, 6.42065,   1.39806e-5),
    (6 // 2, 7.02849,   0.00109846),
    (8 // 2, 9.03102,   0.0000691475),
    (10 // 2, 11.0324,  4.35144e-6),
    (12 // 2, 13.0333,  2.73398e-7),
]


# ---------------------------------------------------------------------------
# Conformal blocks and bootstrap sum
# ---------------------------------------------------------------------------

def Gdiag(delta, n, z_array):
    """Diagonal conformal block G_{delta,n}(z) for an array of z values."""
    z_np = np.atleast_1d(np.asarray(z_array, dtype=np.float64))
    result = np.zeros_like(z_np, dtype=np.float64)

    z_arg = z_np ** 2 / (4 * (z_np - 1))
    prefactor = (-(z_np ** 2 / (z_np - 1))) ** (delta / 2)

    for r in range(n + 1):
        a = [-0.5 + delta / 2, r + delta / 2, delta / 2]
        b = [0.5 + r + delta / 2, -0.5 + delta]
        pFq_val = np.array([float(hyper(a, b, z_val)) for z_val in z_arg])

        coeff = (
            comb(n, r, exact=True)
            * poch(0.5, r)
            * poch(0.5 + n, r)
            * poch((1 + delta) / 2, n)
            * poch(0.5 * (-1 - 2 * n + delta), n - r)
            / (
                poch(delta / 2, n)
                * poch((1 + delta) / 2, r)
                * poch(0.5 * (-2 * n + delta), n)
            )
        )
        result += prefactor * coeff * pFq_val

    return result


def Gsum(z_array):
    """Bootstrap sum G(z) = sum_i a_i * G_{Delta_i, n_i}(z)."""
    z_np = np.atleast_1d(np.asarray(z_array, dtype=np.float64))
    result = np.zeros_like(z_np, dtype=np.float64)
    for n, Delta, a in DATA_BOOTSTRAP:
        result += a * Gdiag(Delta, int(n), z_np)
    return result


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class HNet(nn.Module):
    """
    Network representing H(z) = G(z) - 1.

    The pre-factor encodes the expected singular behaviour at z=0 and z=1.
    The leading power z^1.412625 corresponds to the dimension of the lightest
    operator in the spectrum (epsilon, Delta_epsilon ~ 1.41263).
    """

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return z ** 1.412625 * (1 - z) ** (-2 * self.delta) * self.net(z)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def make_loss_fn(anchor_points, delta):
    """
    Build the loss combining:
      - Residual: crossing equation (z <-> 1-z) symmetry.
      - Anchor: match bootstrap values at selected z points (weight ANCHOR_WEIGHT).
    """
    print("  Computing bootstrap anchor values (this may take a moment)...")
    anchors = [(z, Gsum(np.array([z]))[0]) for z in anchor_points]

    def loss_fn(model, z):
        H_z = model(z)
        H_1mz = model(1 - z)

        g_z = H_z + 1
        g_1mz = H_1mz + 1

        rhs = (z ** (2 * delta) / torch.clamp((1 - z) ** (2 * delta), min=EPS)) * g_1mz
        denom = 1 + torch.abs(g_z) + torch.abs(rhs)
        loss_residual = torch.mean(((g_z - rhs) / denom) ** 2)

        loss_anchor = sum(
            (model(torch.tensor([[zc]], dtype=torch.float64, device=device)) - target) ** 2
            for zc, target in anchors
        )
        return loss_residual + ANCHOR_WEIGHT * loss_anchor, loss_residual, loss_anchor

    return loss_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(seed, z_min, z_max, n_points, max_epochs, patience, anchor_points, delta):
    """Train a single model and return (model, best_loss, final_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HNet(delta).to(device).double()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.98)

    z_train = torch.linspace(z_min, z_max, n_points, dtype=torch.float64, device=device).unsqueeze(1)
    loss_fn = make_loss_fn(anchor_points, delta)

    best_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    final_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        loss, l_resid, l_anchor = loss_fn(model, z_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        for g in optimizer.param_groups:
            g["lr"] = max(g["lr"], 1e-5)

        if loss.item() < best_loss - 1e-8:
            best_loss = loss.item()
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch % 10000 == 0 or epoch == 1:
            print(
                f"    epoch {epoch:6d} | loss {loss.item():.4e} | "
                f"resid {l_resid.item():.4e} | anchor {l_anchor.item():.4e}"
            )

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch} (best loss {best_loss:.4e})")
            final_epoch = epoch
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss, final_epoch


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(z_values, bootstrap_values, all_predictions, output_dir, delta):
    z = np.array(z_values)
    ref = np.array(bootstrap_values)
    preds = np.array(all_predictions)   # shape (n_runs, n_points)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"3D Ising sigma four-point function  (Delta_sigma={delta})", fontsize=13)

    ax = axes[0]
    for i, p in enumerate(preds):
        ax.plot(z, p, color="steelblue", lw=0.6, alpha=0.35,
                label="Individual runs" if i == 0 else None)
    ax.fill_between(z, mean_pred - std_pred, mean_pred + std_pred,
                    color="steelblue", alpha=0.3, label="Mean ± 1σ")
    ax.plot(z, mean_pred, color="steelblue", lw=2, label="Ensemble mean")
    ax.plot(z, ref, "k--", lw=2, label="Bootstrap")
    ax.set_xlabel("z")
    ax.set_ylabel("G(z)")
    ax.set_title("Ensemble vs bootstrap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    rel_err = (mean_pred - ref) / (np.abs(ref) + 1.0) * 100
    rel_std = std_pred / (np.abs(ref) + 1.0) * 100
    ax.plot(z, rel_err, color="firebrick", lw=2, label="Mean error")
    ax.fill_between(z, rel_err - rel_std, rel_err + rel_std,
                    color="firebrick", alpha=0.25, label="±1σ")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("z")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative error (normalised by |G| + 1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ensemble_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_rational(value):
    value = str(value).strip()
    if "/" in value:
        num, denom = value.split("/", 1)
        return float(num) / float(denom)
    return float(value)


def main():
    parser = argparse.ArgumentParser(
        description="Train an ensemble of networks for the 3D Ising sigma four-point function."
    )
    parser.add_argument(
        "--num-runs", type=int, default=5,
        help="Number of independent training runs (default: 5). "
             "Each run is slow due to the conformal block evaluation; start small.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_3d_sigma",
        help="Directory for results and plots (default: results_3d_sigma).",
    )
    parser.add_argument(
        "--delta", type=str, default="0.51815",
        help=f"Conformal dimension of sigma (default: {DELTA_SIGMA}, the 3D Ising value).",
    )
    parser.add_argument(
        "--anchor-points", type=float, nargs="+", default=[0.3],
        help="z values at which the bootstrap answer is enforced (default: 0.3).",
    )
    parser.add_argument("--max-epochs", type=int, default=200000)
    parser.add_argument("--patience",   type=int, default=5000,
                        help="Early-stopping patience in epochs (default: 5000).")
    parser.add_argument("--z-min",      type=float, default=0.01)
    parser.add_argument("--z-max",      type=float, default=0.60)
    parser.add_argument("--n-points",   type=int,   default=600,
                        help="Number of z points in the training grid (default: 600).")
    parser.add_argument("--save-models", action="store_true",
                        help="Save individual model weights as .pt files.")
    args = parser.parse_args()

    delta = parse_rational(args.delta)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("3D Ising Sigma Neural Network Bootstrap")
    print(f"  delta_sigma   : {args.delta} = {delta:.6f}")
    print(f"  Anchor points : {args.anchor_points}")
    print(f"  Runs          : {args.num_runs}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 60)

    print("\nPre-computing bootstrap reference on evaluation grid...")
    z_eval = np.linspace(args.z_min, args.z_max, args.n_points)
    bootstrap_values = (Gsum(z_eval) + 1).tolist()
    print("Bootstrap reference computed.")

    all_predictions = []
    run_metadata = []

    for run_id in range(args.num_runs):
        seed = run_id * 17
        print(f"\n--- Run {run_id + 1}/{args.num_runs}  (seed={seed}) ---")
        t0 = time.time()

        model, best_loss, final_epoch = train_model(
            seed, args.z_min, args.z_max, args.n_points,
            args.max_epochs, args.patience, args.anchor_points, delta,
        )
        elapsed = time.time() - t0

        with torch.no_grad():
            z_t = torch.tensor(z_eval, dtype=torch.float64).unsqueeze(1)
            preds = (model(z_t).cpu().numpy().squeeze() + 1).tolist()

        all_predictions.append(preds)
        run_metadata.append({
            "run_id": run_id,
            "seed": seed,
            "best_loss": best_loss,
            "final_epoch": final_epoch,
            "elapsed_seconds": elapsed,
        })

        result = {**run_metadata[-1], "predictions": preds}
        with open(os.path.join(args.output_dir, f"model_{run_id:03d}.json"), "w") as f:
            json.dump(result, f, indent=2)

        if args.save_models:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"model_{run_id:03d}.pt"))

        print(f"    done in {elapsed:.1f}s | best_loss={best_loss:.4e}")

    metadata = {
        "config": {
            "delta": args.delta,
            "delta_float": delta,
            "anchor_points": args.anchor_points,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "z_min": args.z_min,
            "z_max": args.z_max,
            "n_points": args.n_points,
            "num_runs": args.num_runs,
        },
        "runs": run_metadata,
        "z_values": z_eval.tolist(),
        "bootstrap_values": bootstrap_values,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    losses = [r["best_loss"] for r in run_metadata]
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  Runs completed : {args.num_runs}")
    print(f"  Best loss      : {min(losses):.4e}")
    print(f"  Mean loss      : {np.mean(losses):.4e} ± {np.std(losses):.4e}")
    print("=" * 60)

    plot_results(z_eval.tolist(), bootstrap_values, all_predictions, args.output_dir, args.delta)


if __name__ == "__main__":
    main()
