#!/usr/bin/env python3
"""
Neural network bootstrap for the generalised free boson four-point function.

Trains an ensemble of neural networks to learn the four-point function G(z)
of a scalar operator with conformal dimension Delta in a generalised free
field theory by enforcing the crossing symmetry constraint

    G(z) = (z / (1-z))^(2*Delta) * G(1-z).

The exact answer for the generalised free boson is known analytically:

    G(z) = 1 + z^(2*Delta) + (z/(1-z))^(2*Delta),

and is used only to set the anchor constraint and to assess accuracy.

Usage (personal computer):
    python run_gfb.py --num-runs 10
    python run_gfb.py --num-runs 20 --delta 0.25 --output-dir results_gfb025
    python run_gfb.py --num-runs 5 --delta 1/4 --anchor-points 0.2 0.5

Dependencies: torch, numpy, matplotlib
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


device = torch.device("cpu")
EPS = 1e-8
ANCHOR_WEIGHT = 100.0


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def H_exact(z, delta):
    """H(z) = G(z) - 1 for the generalised free boson."""
    return z ** (2 * delta) + (z / (1 - z)) ** (2 * delta)


def G_exact(z, delta):
    return H_exact(z, delta) + 1


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class HNet(nn.Module):
    """
    Network representing H(z) = G(z) - 1.

    The pre-factor encodes the expected singular behaviour at z=0 and z=1,
    leaving a smooth function for the network to learn.
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
        return z ** (2 * self.delta) * (1 - z) ** (-2 * self.delta) * self.net(z)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def make_loss_fn(anchor_points, delta):
    """
    Build the loss combining:
      - Residual: crossing equation (z <-> 1-z) symmetry.
      - Anchor: match exact values at selected z points (weight ANCHOR_WEIGHT).
    """
    anchor_tensors = [
        (torch.tensor([[z]], dtype=torch.float64, device=device), H_exact(z, delta))
        for z in anchor_points
    ]

    def loss_fn(model, z):
        H_z = model(z)
        H_1mz = model(1 - z)

        g_z = H_z + 1
        g_1mz = H_1mz + 1

        rhs = (z ** (2 * delta) / torch.clamp((1 - z) ** (2 * delta), min=EPS)) * g_1mz
        denom = 1 + torch.abs(g_z) + torch.abs(rhs)
        loss_residual = torch.mean(((g_z - rhs) / denom) ** 2)

        loss_anchor = sum(
            (model(z_t) - target) ** 2 for z_t, target in anchor_tensors
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

def plot_results(z_values, exact_values, all_predictions, output_dir, delta):
    z = np.array(z_values)
    exact = np.array(exact_values)
    preds = np.array(all_predictions)   # shape (n_runs, n_points)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Generalised Free Boson  Delta={delta}", fontsize=13)

    # Panel 1: spaghetti + mean band vs exact
    ax = axes[0]
    for i, p in enumerate(preds):
        ax.plot(z, p, color="steelblue", lw=0.6, alpha=0.35,
                label="Individual runs" if i == 0 else None)
    ax.fill_between(z, mean_pred - std_pred, mean_pred + std_pred,
                    color="steelblue", alpha=0.3, label="Mean ± 1σ")
    ax.plot(z, mean_pred, color="steelblue", lw=2, label="Ensemble mean")
    ax.plot(z, exact, "k--", lw=2, label="Exact")
    ax.set_xlabel("z")
    ax.set_ylabel("G(z)")
    ax.set_title("Ensemble vs exact")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: relative error
    ax = axes[1]
    rel_err = (mean_pred - exact) / (np.abs(exact) + 1.0) * 100
    rel_std = std_pred / (np.abs(exact) + 1.0) * 100
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
        description="Train an ensemble of networks for the generalised free boson four-point function."
    )
    parser.add_argument(
        "--num-runs", type=int, default=10,
        help="Number of independent training runs (default: 10).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_gfb",
        help="Directory for results and plots (default: results_gfb).",
    )
    parser.add_argument(
        "--delta", type=str, default="0.5",
        help="Conformal dimension of the fundamental scalar. "
             "Supports fractions like '1/4' or decimals (default: 0.5).",
    )
    parser.add_argument(
        "--anchor-points", type=float, nargs="+", default=[0.3],
        help="z values at which the exact answer is enforced (default: 0.3).",
    )
    parser.add_argument("--max-epochs", type=int, default=200000)
    parser.add_argument("--patience",   type=int, default=5000,
                        help="Early-stopping patience in epochs (default: 5000).")
    parser.add_argument("--z-min",      type=float, default=0.01)
    parser.add_argument("--z-max",      type=float, default=0.90)
    parser.add_argument("--n-points",   type=int,   default=1000,
                        help="Number of z points in the training grid (default: 1000).")
    parser.add_argument("--save-models", action="store_true",
                        help="Save individual model weights as .pt files.")
    args = parser.parse_args()

    delta = parse_rational(args.delta)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generalised Free Boson Neural Network Bootstrap")
    print(f"  delta         : {args.delta} = {delta:.6f}")
    print(f"  Anchor points : {args.anchor_points}")
    print(f"  Runs          : {args.num_runs}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 60)

    z_eval = np.linspace(args.z_min, args.z_max, args.n_points)
    exact_values = G_exact(z_eval, delta).tolist()

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

    # Save shared metadata
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
        "exact_values": exact_values,
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

    plot_results(z_eval.tolist(), exact_values, all_predictions, args.output_dir, args.delta)


if __name__ == "__main__":
    main()
