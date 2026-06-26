#!/usr/bin/env python3
"""
Neural network bootstrap for torus partition functions of A/D/E Virasoro minimal models.

For a chosen modular invariant (series A, D, or E at level m), builds the
reduced correlator G(z) = |z^2 / (2^8 (1-z))|^(c/12) Z(tau(z)), then trains
an ensemble of `alog_alg` ansatz networks that enforce:

  G(z) ≈ G_exact(z)

via an anchor condition at a reference point z=anchor_z.  The crossing
symmetry of the torus partition function is automatically encoded in Z; the
networks learn the z-dependence of the prefactor.

Supported (series, m) pairs
----------------------------
  A  : any m >= 3   (diagonal A-series, d_gap = 3 / (m(m+1)))
  D  : any m >= 5 with p or q even  (d_gap = 8 / (m(m+1)))
  E6 : m in {11, 12}   (d_gap = 15 / (m(m+1)))
  E7 : m in {17, 18}   (d_gap =  8 / (m(m+1)))
  E8 : m in {29, 30}   (d_gap = 48 / (m(m+1)))

Usage
-----
  python run_torus_ade.py --series A --m 4
  python run_torus_ade.py --series D --m 6 --n-seeds 5 --output-dir results_D6
  python run_torus_ade.py --series E --m 11 --n-seeds 10 --max-epochs 300000
  python run_torus_ade.py --series A --m 3 --log-every 0  # silence per-epoch prints

Dependencies: torch, numpy, scipy, matplotlib
"""

import argparse
import cmath
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk
import torch
import torch.nn as nn
import torch.optim as optim


EPS = 1e-8

# ---------------------------------------------------------------------------
# Minimal-model helpers
# ---------------------------------------------------------------------------

def central_charge(p, q):
    return 1.0 - 6.0 * (p - q) ** 2 / (p * q)


def primary_labels(p, q):
    """One representative per Kac identification (r,s) ~ (p-r, q-s)."""
    seen, out = set(), []
    for r in range(1, p):
        for s in range(1, q):
            canon = min((r, s), (p - r, q - s))
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    return out


def tau_from_cross_ratio(z):
    return 1j * ellipk(1.0 - z) / ellipk(z)


def dedekind_eta(tau, terms=200):
    q = cmath.exp(2j * math.pi * tau)
    prod = 1.0 + 0.0j
    qn = q
    for _ in range(terms):
        prod *= (1.0 - qn)
        qn *= q
    return q ** (1.0 / 24.0) * prod


def character(p, q, r, s, tau, theta_terms=80, eta_terms=200):
    """Rocha-Caridi Virasoro minimal-model character."""
    qt = cmath.exp(2j * math.pi * tau)
    total = 0.0 + 0.0j
    for n in range(-theta_terms, theta_terms + 1):
        L = (2 * p * q * n + q * r - p * s) ** 2 / (4.0 * p * q)
        R = (2 * p * q * n + q * r + p * s) ** 2 / (4.0 * p * q)
        total += qt ** L - qt ** R
    return total / dedekind_eta(tau, terms=eta_terms)


def compute_B(z_array, n_eta_terms=50):
    """B(z) = sqrt(tau_2) * eta(i*tau_2)^2.

    Uses the modular identity eta(i*y) = eta(i/y)/sqrt(y) when y < 1 so
    that the q-product always converges.
    """
    z = np.asarray(z_array, dtype=np.float64)
    y = ellipk(1.0 - z) / ellipk(z)
    sqrt_y = np.sqrt(y)
    y_eval = np.where(y < 1.0, 1.0 / y, y)
    q = np.exp(-2.0 * np.pi * y_eval)
    log_eta = -np.pi * y_eval / 12.0
    for n in range(1, n_eta_terms + 1):
        log_eta = log_eta + np.log(1.0 - q ** n)
    eta_eval = np.exp(log_eta)
    eta_y = np.where(y < 1.0, eta_eval / sqrt_y, eta_eval)
    return sqrt_y * eta_y ** 2


# ---------------------------------------------------------------------------
# Build modular invariant Z(tau) for the chosen (series, m)
# ---------------------------------------------------------------------------

def build_theory(series, m):
    """Return (Z_target_fn, series_label, d_gap, c, p, q) for the given series/m.

    Z_target_fn(tau) evaluates the modular-invariant torus partition function at
    modular parameter tau (a complex number).
    """
    series = series.upper()
    p, q = m, m + 1
    c = central_charge(p, q)

    if series == 'A':
        if m < 3:
            raise ValueError(f"A-series requires m >= 3; got m={m}.")
        d_gap = 3.0 / (m * (m + 1))

        def Z_target(tau):
            total = 0.0
            for r, s in primary_labels(p, q):
                chi = character(p, q, r, s, tau)
                total += abs(chi) ** 2
            return float(total)

        series_label = f'A-series  M({p},{q})'

    elif series == 'D':
        d_gap = 8.0 / (m * (m + 1))
        if q % 2 == 0:
            N_even, side = q, 'q'
        elif p % 2 == 0:
            N_even, side = p, 'p'
        else:
            raise ValueError(
                f"D-series needs p or q even; M({p},{q}) has both odd. "
                "Try m=5,6,8,10,12,13,14,15,..."
            )
        half = N_even // 2
        if half < 3:
            raise ValueError(
                f"D-series is degenerate (= A-series) for M({p},{q}); pick m >= 5."
            )
        odd_labels = list(range(1, N_even, 2))
        pairs = [(k, N_even - k) for k in odd_labels if k < half]
        fixed = half if (half % 2 == 1) else None
        D_index = half + 1

        def Z_target(tau):
            total = 0.0
            outer_range = range(1, p) if side == 'q' else range(1, q)
            for outer in outer_range:
                acc = 0.0
                for k1, k2 in pairs:
                    if side == 'q':
                        c1 = character(p, q, outer, k1, tau)
                        c2 = character(p, q, outer, k2, tau)
                    else:
                        c1 = character(p, q, k1, outer, tau)
                        c2 = character(p, q, k2, outer, tau)
                    acc += abs(c1 + c2) ** 2
                if fixed is not None:
                    cf = (character(p, q, outer, fixed, tau) if side == 'q'
                          else character(p, q, fixed, outer, tau))
                    acc += 2 * abs(cf) ** 2
                total += 0.5 * acc
            return float(total)

        series_label = f'D{D_index} ({side}-side)  M({p},{q})'

    elif series == 'E':
        if (p, q) == (11, 12):
            d_gap = 15.0 / (m * (m + 1))

            def Z_target(tau):
                total = 0.0
                for r in range(1, p):
                    c1  = character(p, q, r, 1,  tau)
                    c7  = character(p, q, r, 7,  tau)
                    c4  = character(p, q, r, 4,  tau)
                    c8  = character(p, q, r, 8,  tau)
                    c5  = character(p, q, r, 5,  tau)
                    c11 = character(p, q, r, 11, tau)
                    total += 0.5 * (abs(c1 + c7) ** 2 + abs(c4 + c8) ** 2 + abs(c5 + c11) ** 2)
                return float(total)

            series_label = 'E6 (q-side)  M(11,12)'

        elif (p, q) == (12, 13):
            d_gap = 15.0 / (m * (m + 1))

            def Z_target(tau):
                total = 0.0
                for s in range(1, q):
                    c1  = character(p, q, 1,  s, tau)
                    c7  = character(p, q, 7,  s, tau)
                    c4  = character(p, q, 4,  s, tau)
                    c8  = character(p, q, 8,  s, tau)
                    c5  = character(p, q, 5,  s, tau)
                    c11 = character(p, q, 11, s, tau)
                    total += 0.5 * (abs(c1 + c7) ** 2 + abs(c4 + c8) ** 2 + abs(c5 + c11) ** 2)
                return float(total)

            series_label = 'E6 (p-side)  M(12,13)'

        elif (p, q) == (17, 18):
            d_gap = 8.0 / (m * (m + 1))

            def Z_target(tau):
                total = 0.0
                for r in range(1, p):
                    c1  = character(p, q, r, 1,  tau)
                    c17 = character(p, q, r, 17, tau)
                    c5  = character(p, q, r, 5,  tau)
                    c13 = character(p, q, r, 13, tau)
                    c7  = character(p, q, r, 7,  tau)
                    c11 = character(p, q, r, 11, tau)
                    c9  = character(p, q, r, 9,  tau)
                    c3  = character(p, q, r, 3,  tau)
                    c15 = character(p, q, r, 15, tau)
                    off  = 2.0 * (c9 * (c3 + c15).conjugate()).real
                    diag = abs(c1 + c17) ** 2 + abs(c5 + c13) ** 2 + abs(c7 + c11) ** 2 + abs(c9) ** 2
                    total += 0.5 * (diag + off)
                return float(total)

            series_label = 'E7 (q-side)  M(17,18)'

        elif (p, q) == (18, 19):
            d_gap = 8.0 / (m * (m + 1))

            def Z_target(tau):
                total = 0.0
                for s in range(1, q):
                    c1  = character(p, q, 1,  s, tau)
                    c17 = character(p, q, 17, s, tau)
                    c5  = character(p, q, 5,  s, tau)
                    c13 = character(p, q, 13, s, tau)
                    c7  = character(p, q, 7,  s, tau)
                    c11 = character(p, q, 11, s, tau)
                    c9  = character(p, q, 9,  s, tau)
                    c3  = character(p, q, 3,  s, tau)
                    c15 = character(p, q, 15, s, tau)
                    off  = 2.0 * (c9 * (c3 + c15).conjugate()).real
                    diag = abs(c1 + c17) ** 2 + abs(c5 + c13) ** 2 + abs(c7 + c11) ** 2 + abs(c9) ** 2
                    total += 0.5 * (diag + off)
                return float(total)

            series_label = 'E7 (p-side)  M(18,19)'

        elif (p, q) == (29, 30):
            d_gap = 48.0 / (m * (m + 1))

            def Z_target(tau):
                total = 0.0
                for r in range(1, p):
                    c1  = character(p, q, r, 1,  tau)
                    c11 = character(p, q, r, 11, tau)
                    c19 = character(p, q, r, 19, tau)
                    c29 = character(p, q, r, 29, tau)
                    c7  = character(p, q, r, 7,  tau)
                    c13 = character(p, q, r, 13, tau)
                    c17 = character(p, q, r, 17, tau)
                    c23 = character(p, q, r, 23, tau)
                    total += 0.5 * (abs(c1 + c11 + c19 + c29) ** 2 + abs(c7 + c13 + c17 + c23) ** 2)
                return float(total)

            series_label = 'E8 (q-side)  M(29,30)'

        elif (p, q) == (30, 31):
            d_gap = 48.0 / (m * (m + 1))

            def Z_target(tau):
                total = 0.0
                for s in range(1, q):
                    c1  = character(p, q, 1,  s, tau)
                    c11 = character(p, q, 11, s, tau)
                    c19 = character(p, q, 19, s, tau)
                    c29 = character(p, q, 29, s, tau)
                    c7  = character(p, q, 7,  s, tau)
                    c13 = character(p, q, 13, s, tau)
                    c17 = character(p, q, 17, s, tau)
                    c23 = character(p, q, 23, s, tau)
                    total += 0.5 * (abs(c1 + c11 + c19 + c29) ** 2 + abs(c7 + c13 + c17 + c23) ** 2)
                return float(total)

            series_label = 'E8 (p-side)  M(30,31)'

        else:
            raise ValueError(
                f"E-series only exists for m in {{11,12}} (E6), {{17,18}} (E7), "
                f"{{29,30}} (E8); got m={m}."
            )

    else:
        raise ValueError(f"Unknown series {series!r}. Choose A, D, or E.")

    return Z_target, series_label, d_gap, c, p, q


# ---------------------------------------------------------------------------
# Reduced correlator G(z)
# ---------------------------------------------------------------------------

def make_G_target(Z_target_fn, c):
    """Return a scalar function G(z) = |z^2 / (2^8(1-z))|^(c/12) * Z(tau(z))."""
    def G_target(z):
        tau = tau_from_cross_ratio(float(z))
        return abs(float(z) ** 2 / (2 ** 8 * (1.0 - float(z)))) ** (c / 12.0) * Z_target_fn(tau)
    return G_target


# ---------------------------------------------------------------------------
# alog_alg ansatz network
# ---------------------------------------------------------------------------

class ANet(nn.Module):
    """
    alog_alg ansatz:
        G(z) ≈ c0 * z^p0 * sqrt(log(16/z))  +  z^p_corr * (1-z)^p1 * NN(z)

    The leading term captures the OPE singularity; the correction encodes
    the gap-dependent subleading behaviour.
    """

    def __init__(self, c0_const, p0, p_corr, p1, width=64):
        super().__init__()
        self.c0_const = c0_const
        self.p0 = p0
        self.p_corr = p_corr
        self.p1 = p1
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, z):
        leading    = self.c0_const * z.pow(self.p0) * torch.sqrt(torch.log(16.0 / z))
        correction = z.pow(self.p_corr) * (1.0 - z).pow(self.p1) * self.net(z)
        return leading + correction


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(seed, z_grid, G_exact, B_z, anchors,
              c, d_gap, width, max_epochs, patience, lr, log_every):
    """Train one seed of the alog_alg ansatz.

    anchors: list of (z_val, G_val, B_val) tuples — anchor points where the
             exact reduced correlator B*G is enforced.

    Returns a dict: seed, final_epoch, best_loss, elapsed, G_pred, max_err,
    early_stopped, model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    c0_const = (16.0 ** (-1.0 / 6.0)) / np.sqrt(np.pi)
    p0       = 1.0 / 6.0
    p1       = (2.0 - 3.0 * c) / 12.0
    p_corr   = d_gap + 1.0 / 6.0

    model = ANet(c0_const, p0, p_corr, p1, width=width).double()
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.98)

    z_t     = torch.tensor(z_grid, dtype=torch.float64).unsqueeze(1)
    B_z_t   = torch.tensor(B_z, dtype=torch.float64).unsqueeze(1)
    G_exact_t = torch.tensor(G_exact, dtype=torch.float64).unsqueeze(1)
    anchor_tensors = [
        (torch.tensor([[az]], dtype=torch.float64),
         torch.tensor([[bz * gz]], dtype=torch.float64))
        for az, gz, bz in anchors
    ]

    if log_every > 0:
        print(f"[seed {seed:3d}] starting:  max_epochs={max_epochs}  patience={patience}  lr={lr}")
        print(f"[seed {seed:3d}] columns:  epoch | total_loss | cross | anch | best_loss | max_G_err | elapsed")

    best_loss = float('inf')
    best_state = None
    no_imp = 0
    final_ep = max_epochs
    early_stopped = False
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        opt.zero_grad()
        g    = model(z_t)
        g_1m = model(1.0 - z_t)
        cf   = (z_t / torch.clamp(1.0 - z_t, min=EPS)) ** (c / 4.0)
        denom = 1.0 + torch.abs(g) + torch.abs(cf * g_1m)
        loss_cross = torch.mean(((g - cf * g_1m) / denom) ** 2)
        loss_anch  = sum((model(az_t) - gred_t) ** 2 for az_t, gred_t in anchor_tensors)
        loss = loss_cross + 100.0 * loss_anch
        loss.backward()
        opt.step()
        sched.step()

        lv = float(loss.item())
        if lv < best_loss - 1e-10:
            best_loss = lv
            no_imp = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if log_every > 0 and (ep % log_every == 0 or ep == 1):
            with torch.no_grad():
                err_now = float((g / B_z_t - G_exact_t).abs().max())
            print(
                f"[seed {seed:3d}] ep={ep:>6d}  loss={lv:.2e}  "
                f"cross={loss_cross.detach().item():.2e}  anch={loss_anch.detach().item():.2e}  "
                f"best={best_loss:.2e}  max_err={err_now:.3e}  t={time.time()-t0:.1f}s"
            )

        if no_imp >= patience:
            final_ep = ep
            early_stopped = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    with torch.no_grad():
        g_pred = model(z_t).cpu().numpy().squeeze() / B_z

    max_err = float(np.abs(g_pred - G_exact).max())

    if log_every > 0:
        tag = "EARLY-STOP" if early_stopped else "MAX-EPOCHS"
        print(
            f"[seed {seed:3d}] DONE ({tag}):  final_epoch={final_ep}  "
            f"best_loss={best_loss:.3e}  max_G_err={max_err:.3e}  elapsed={elapsed:.1f}s"
        )

    return {
        'seed': seed,
        'final_epoch': final_ep,
        'best_loss': best_loss,
        'elapsed': elapsed,
        'G_pred': g_pred,
        'max_err': max_err,
        'early_stopped': early_stopped,
        'model': model,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_example(z_grid, B_z, G_exact, example_run, series_label, c, output_dir):
    """Two-panel plot for a single seed: predicted vs exact and residuals."""
    G_red_exact = B_z * G_exact
    G_red_pred  = B_z * example_run['G_pred']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(z_grid, G_red_exact, 'k--', lw=1.6, label='Exact')
    ax.plot(z_grid, G_red_pred, color='tab:blue', lw=1.3,
            label=f"seed={example_run['seed']}")
    ax.set_xlabel('z')
    ax.set_ylabel(r'$G_{\rm red}(z) = B(z)\,G(z)$')
    ax.set_title(f'{series_label} — single seed (c={c:.4f})')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0, color='k', lw=0.6)
    ax.plot(
        z_grid,
        100.0 * (G_red_pred - G_red_exact) / (1.0 + np.abs(G_red_exact)),
        color='tab:blue', lw=1.3,
    )
    ax.set_xlabel('z')
    ax.set_ylabel('error (%)')
    ax.set_title('Scale-normalized % error')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'example_seed.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Example seed plot saved to {path}")


def plot_spaghetti(z_grid, B_z, G_exact, runs, series_label, c, d_gap, output_dir):
    """Two-panel spaghetti plot: all seeds overlaid on exact, plus per-seed errors.

    Matches the style of run_2d_minimal_models.py: individual seed curves shown
    thin and semi-transparent, with the ensemble mean and ±1σ band overlaid in
    both panels.
    """
    Gred_exact = B_z * G_exact
    Gred_preds = np.array([B_z * r['G_pred'] for r in runs])
    mean_pred  = Gred_preds.mean(axis=0)
    std_pred   = Gred_preds.std(axis=0)

    errs = 100.0 * (Gred_preds - Gred_exact) / (1.0 + np.abs(Gred_exact))
    mean_err = errs.mean(axis=0)
    std_err  = errs.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Torus alog_alg — {series_label}  (c={c:.4f},  d_gap={d_gap:.4f})',
        fontsize=13,
    )

    ax = axes[0]
    for i, r in enumerate(runs):
        ax.plot(z_grid, B_z * r['G_pred'], color='steelblue', lw=0.6, alpha=0.35,
                label='Individual seeds' if i == 0 else None)
    ax.fill_between(z_grid, mean_pred - std_pred, mean_pred + std_pred,
                    color='steelblue', alpha=0.3, label='Mean ± 1σ')
    ax.plot(z_grid, mean_pred, color='steelblue', lw=2, label='Ensemble mean')
    ax.plot(z_grid, Gred_exact, 'k--', lw=1.8, label='Exact')
    ax.set_xlabel('z')
    ax.set_ylabel(r'$G_{\rm red}(z)$')
    ax.set_title(f'Ensemble vs exact  ({len(runs)} seeds)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.fill_between(z_grid, mean_err - std_err, mean_err + std_err,
                    color='firebrick', alpha=0.25, label='±1σ')
    ax.plot(z_grid, mean_err, color='firebrick', lw=2, label='Mean error')
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('z')
    ax.set_ylabel('error (%)')
    ax.set_title('Scale-normalized % error (normalised by $|G_{\\rm red}|$ + 1)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'spaghetti.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spaghetti plot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train an ensemble of alog_alg networks for torus partition "
                    "functions of A/D/E minimal models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--series', type=str, default='A', choices=['A', 'D', 'E'],
        help="Modular-invariant series: A (diagonal), D, or E (exceptional). Default: A.",
    )
    parser.add_argument(
        '--m', type=int, default=4,
        help="Minimal-model index m (p=m, q=m+1). "
             "A: m>=3; D: m>=5 with p or q even; E: m in {11,12,17,18,29,30}. Default: 4.",
    )
    parser.add_argument(
        '--n-seeds', '--num-runs', type=int, default=10, dest='n_seeds',
        help="Number of independent training seeds. Default: 10.",
    )
    parser.add_argument(
        '--output-dir', type=str, default='results_torus_ade',
        help="Directory for plots, per-seed JSON, and metadata. Default: results_torus_ade.",
    )
    parser.add_argument(
        '--max-epochs', type=int, default=200_000,
        help="Maximum training epochs per seed. Default: 200000.",
    )
    parser.add_argument(
        '--patience', type=int, default=5_000,
        help="Early-stopping patience in epochs. Default: 5000.",
    )
    parser.add_argument(
        '--lr', type=float, default=5e-4,
        help="Initial Adam learning rate. Default: 5e-4.",
    )
    parser.add_argument(
        '--width', type=int, default=64,
        help="Hidden-layer width of ANet. Default: 64.",
    )
    parser.add_argument(
        '--n-points', type=int, default=90,
        help="Number of z points in the training/evaluation grid. Default: 90.",
    )
    parser.add_argument(
        '--z-min', type=float, default=0.01,
        help="Left endpoint of the z grid. Default: 0.01.",
    )
    parser.add_argument(
        '--z-max', type=float, default=0.90,
        help="Right endpoint of the z grid. Default: 0.90.",
    )
    parser.add_argument(
        '--anchor-points', '--anchor-z', type=float, nargs='+', default=[0.4],
        dest='anchor_points',
        help="One or more z values at which the exact answer is enforced. Default: 0.4.",
    )
    parser.add_argument(
        '--log-every', type=int, default=20_000,
        help="Print progress every this many epochs (0 = silent). Default: 20000.",
    )
    parser.add_argument(
        '--example-log-every', type=int, default=5_000,
        help="Progress cadence for the first (example) seed only. Default: 5000.",
    )
    parser.add_argument(
        '--save-models', action='store_true',
        help="Save each seed's model weights as seed_NNN.pt in output-dir.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build theory
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Torus ADE  —  Neural-network alog_alg bootstrap")
    try:
        Z_target_fn, series_label, d_gap, c, p, q = build_theory(args.series, args.m)
    except ValueError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)

    c0_const = (16.0 ** (-1.0 / 6.0)) / np.sqrt(np.pi)
    p0       = 1.0 / 6.0
    p1       = (2.0 - 3.0 * c) / 12.0
    p_corr   = d_gap + 1.0 / 6.0

    print(f"  Theory     : {series_label}")
    print(f"  c          : {c:.6f}")
    print(f"  d_gap      : {d_gap:.6f}")
    print(f"  Ansatz exponents:  p0={p0:.4f}  p_corr={p_corr:.4f}  p1={p1:.4f}")
    print(f"  Seeds      : {args.n_seeds}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build z-grid and exact target
    # ------------------------------------------------------------------
    print("Building z-grid and evaluating exact target  ", end='', flush=True)
    t_grid = time.time()
    G_target_fn = make_G_target(Z_target_fn, c)
    z_grid  = np.linspace(args.z_min, args.z_max, args.n_points, dtype=np.float64)
    G_exact = np.array([G_target_fn(zi) for zi in z_grid])
    B_z     = compute_B(z_grid)
    anchors = [
        (az, G_target_fn(az), float(compute_B(np.array([az]))[0]))
        for az in args.anchor_points
    ]
    print(f"done in {time.time()-t_grid:.1f}s")
    print(f"  z range       : [{z_grid[0]:.3f}, {z_grid[-1]:.3f}]   n={len(z_grid)}")
    print(f"  G_exact       : [{G_exact.min():.4f}, {G_exact.max():.4f}]")
    print(f"  anchor points : {args.anchor_points}")

    # ------------------------------------------------------------------
    # Train seeds
    # ------------------------------------------------------------------
    runs = []
    train_kwargs = dict(
        z_grid=z_grid, G_exact=G_exact, B_z=B_z,
        anchors=anchors,
        c=c, d_gap=d_gap, width=args.width,
        max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
    )

    for seed in range(args.n_seeds):
        log_ev = args.example_log_every if seed == 0 else args.log_every
        print(f"\n==== seed {seed}/{args.n_seeds - 1} ====")
        result = train_one(seed=seed, log_every=log_ev, **train_kwargs)
        runs.append(result)

        # Save per-seed JSON (G_pred as list; model object excluded)
        record = {k: v for k, v in result.items() if k not in ('G_pred', 'model')}
        record['G_pred'] = result['G_pred'].tolist()
        with open(os.path.join(args.output_dir, f'seed_{seed:03d}.json'), 'w') as f:
            json.dump(record, f, indent=2)

        if args.save_models:
            pt_path = os.path.join(args.output_dir, f'seed_{seed:03d}.pt')
            torch.save(result['model'].state_dict(), pt_path)
            print(f"  Model weights saved to {pt_path}")

    # ------------------------------------------------------------------
    # Ensemble summary
    # ------------------------------------------------------------------
    max_errs = np.array([r['max_err'] for r in runs])
    losses   = np.array([r['best_loss'] for r in runs])
    epochs   = np.array([r['final_epoch'] for r in runs])
    n_es     = sum(1 for r in runs if r['early_stopped'])

    print(f"\n{'=' * 65}")
    print(f"ENSEMBLE SUMMARY  (n={len(runs)})")
    print(f"  early-stopped : {n_es}/{len(runs)}")
    print(f"  best_loss     : min={losses.min():.2e}  "
          f"median={np.median(losses):.2e}  max={losses.max():.2e}")
    print(f"  max_G_err     : min={max_errs.min():.2e}  "
          f"median={np.median(max_errs):.2e}  max={max_errs.max():.2e}")
    print(f"  final_epoch   : min={epochs.min()}  "
          f"median={int(np.median(epochs))}  max={epochs.max()}")

    i05 = int(np.argmin(np.abs(z_grid - 0.5)))
    Gred_exact  = B_z * G_exact
    Gred_preds  = np.array([B_z * r['G_pred'] for r in runs])
    vals = Gred_preds[:, i05]
    print(f"\n  At z=0.5:")
    print(f"    exact G_red = {Gred_exact[i05]:.5f}")
    print(f"    mean        = {vals.mean():.5f} ± {vals.std():.5f}  (n={len(runs)})")
    print(f"{'=' * 65}")

    # Save global metadata
    metadata = {
        'config': {
            'series': args.series, 'm': args.m,
            'series_label': series_label,
            'c': c, 'd_gap': d_gap,
            'n_seeds': args.n_seeds,
            'max_epochs': args.max_epochs, 'patience': args.patience,
            'lr': args.lr, 'width': args.width,
            'z_min': args.z_min, 'z_max': args.z_max, 'n_points': args.n_points,
            'anchor_points': args.anchor_points,
        },
        'z_grid': z_grid.tolist(),
        'G_exact': G_exact.tolist(),
        'B_z': B_z.tolist(),
        'runs': [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items() if k != 'model'}
            for r in runs
        ],
    }
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_example(z_grid, B_z, G_exact, runs[0], series_label, c, args.output_dir)
    plot_spaghetti(z_grid, B_z, G_exact, runs, series_label, c, d_gap, args.output_dir)


if __name__ == '__main__':
    main()
