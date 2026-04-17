#!/usr/bin/env python3
"""
Neural network bootstrap for the AdS2 contact diagram (phi^4, Delta_phi=1).

Trains an ensemble of neural networks to learn the four-point function G(z)
of a scalar with conformal dimension Delta_phi=1 in AdS2, arising from a
phi^4 contact interaction. The function G(z) = H(z) + L(z) is decomposed
into a known leading contribution L(z) and a remainder H(z) that the network learns. The
crossing symmetry constraint reads

    G(z) = (z / (1-z))^2 * G(1-z),

with L(z) = 2 z^2 (log(z) - 1). The exact answer is known analytically and
is used only to set the anchor constraint and to assess accuracy.

Usage (personal computer):
    python run_ads2_contact.py --num-runs 10
    python run_ads2_contact.py --num-runs 20 --output-dir results_ads2
    python run_ads2_contact.py --num-runs 5 --anchor-points 0.3 0.5

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
ANCHOR_WEIGHT = 200.0


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def L_z(z):
    """Known leading contribution: L(z) = 2 z^2 (log(z) - 1)."""
    zc = torch.clamp(z, min=EPS)
    return 2 * zc**2 * (torch.log(zc) - 1)


def H_exact(z):
    """Exact remainder H(z)."""
    zc = torch.clamp(z, min=EPS)
    return 2 * zc**2 * (torch.log(1 - zc) / zc + torch.log(zc) / (1 - zc)) - 2 * zc**2 * (-1 + torch.log(zc))


def G_exact(z):
    """Exact four-point function G(z) = H(z) + L(z)."""
    return H_exact(z) + L_z(z)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class HNet(nn.Module):
    """
    Network representing the crossing-symmetric remainder H(z).

    The pre-factor (z^3 log(z) + z^3 log(1-z)) encodes the expected analytic
    structure, leaving a smooth function for the network to learn.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        zc = torch.clamp(z, min=EPS)
        return (zc**3 * torch.log(zc) + zc**3 * torch.log(1 - zc)) * self.net(z)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def make_loss_fn(anchor_points):
    """
    Build the loss combining:
      - Residual: crossing equation G(z) = (z/(1-z))^2 * G(1-z).
      - Anchor: match exact values at selected z points (weight ANCHOR_WEIGHT).
    """
    anchor_tensors = [
        (
            torch.tensor([[z]], dtype=torch.float64, device=device),
            H_exact(torch.tensor(z, dtype=torch.float64)),
        )
        for z in anchor_points
    ]

    def loss_fn(model, z):
        H_z = model(z)
        H_1mz = model(1 - z)

        G_z = H_z + L_z(z)
        G_1mz = H_1mz + L_z(1 - z)

        rhs = (z**2 / torch.clamp((1 - z)**2, min=EPS)) * G_1mz
        denom = 1 + torch.abs(G_z) + torch.abs(rhs)
        loss_residual = torch.mean(((G_z - rhs) / denom) ** 2)

        loss_anchor = sum(
            (model(z_t) - target) ** 2 for z_t, target in anchor_tensors
        )
        return loss_residual + ANCHOR_WEIGHT * loss_anchor, loss_residual, loss_anchor

    return loss_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(seed, z_min, z_max, n_points, max_epochs, patience, anchor_points):
    """Train a single model and return (model, best_loss, final_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HNet().to(device).double()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.98)

    z_train = torch.linspace(z_min, z_max, n_points, dtype=torch.float64, device=device).unsqueeze(1)
    loss_fn = make_loss_fn(anchor_points)

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
            g["lr"] = max(g["lr"], 1e-6)

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

def plot_results(z_values, exact_values, all_predictions, output_dir):
    z = np.array(z_values)
    exact = np.array(exact_values)
    preds = np.array(all_predictions)   # shape (n_runs, n_points)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AdS2 contact diagram  (phi^4, Delta_phi=1)", fontsize=13)

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

def main():
    parser = argparse.ArgumentParser(
        description="Train an ensemble of networks for the AdS2 contact diagram."
    )
    parser.add_argument(
        "--num-runs", type=int, default=10,
        help="Number of independent training runs (default: 10).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_ads2_contact",
        help="Directory for results and plots (default: results_ads2_contact).",
    )
    parser.add_argument(
        "--anchor-points", type=float, nargs="+", default=[0.4],
        help="z values at which the exact answer is enforced (default: 0.4).",
    )
    parser.add_argument("--max-epochs", type=int, default=200000)
    parser.add_argument("--patience",   type=int, default=5000,
                        help="Early-stopping patience in epochs (default: 5000).")
    parser.add_argument("--z-min",      type=float, default=0.01)
    parser.add_argument("--z-max",      type=float, default=0.90)
    parser.add_argument("--n-points",   type=int,   default=600,
                        help="Number of z points in the training grid (default: 600).")
    parser.add_argument("--save-models", action="store_true",
                        help="Save individual model weights as .pt files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("AdS2 Contact Diagram Neural Network Bootstrap")
    print(f"  Delta_phi     : 1 (fixed)")
    print(f"  Anchor points : {args.anchor_points}")
    print(f"  Runs          : {args.num_runs}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 60)

    z_eval = np.linspace(args.z_min, args.z_max, args.n_points)
    with torch.no_grad():
        z_t_meta = torch.tensor(z_eval, dtype=torch.float64).unsqueeze(1)
        exact_values = G_exact(z_t_meta).cpu().numpy().squeeze().tolist()

    all_predictions = []
    run_metadata = []

    for run_id in range(args.num_runs):
        seed = run_id * 17
        print(f"\n--- Run {run_id + 1}/{args.num_runs}  (seed={seed}) ---")
        t0 = time.time()

        model, best_loss, final_epoch = train_model(
            seed, args.z_min, args.z_max, args.n_points,
            args.max_epochs, args.patience, args.anchor_points,
        )
        elapsed = time.time() - t0

        with torch.no_grad():
            z_t = torch.tensor(z_eval, dtype=torch.float64).unsqueeze(1)
            preds = (model(z_t) + L_z(z_t)).cpu().numpy().squeeze().tolist()

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
            "delta_phi": 1,
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

    plot_results(z_eval.tolist(), exact_values, all_predictions, args.output_dir)


if __name__ == "__main__":
    main()
