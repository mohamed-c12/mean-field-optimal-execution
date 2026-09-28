from pathlib import Path
import csv
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

root = Path(__file__).resolve().parents[1]
rows = []

for seed in (0, 1, 42):
    print(f"\nEntrainement avec la graine {seed}...", flush=True)
    environment = os.environ.copy()
    environment["PINN_SEED"] = str(seed)

    process = subprocess.run(
        [sys.executable, str(root / "src" / "pinn_hjb.py")],
        env=environment,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError(f"Echec pour la graine {seed}")

    metrics_path = root / "results" / f"pinn_hjb_metrics_seed_{seed}.json"
    metrics = json.loads(metrics_path.read_text())
    checkpoint = torch.load(
        root / "models" / f"pinn_hjb_seed_{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1), nn.Softplus(),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    T = checkpoint["T"]
    eta = checkpoint["eta"]
    rho = checkpoint["rho"]
    q0 = 1000.0
    K = np.sqrt(rho * T**2 / eta)
    exact_cost = eta / T * K / np.tanh(K) * q0**2

    # s = remaining time / T.
    # z = q/s removes the terminal singularity.
    # Integrate backwards from s=1 to s=0.
    def dynamics(s, state):
        z = state[0]
        with torch.no_grad():
            f = model(torch.tensor([[s]])).item()

        q = s * z
        v = z / T * (1 + s**2 * f)

        x = K * s
        if abs(x) < 1e-4:
            exact_factor = 1 + x**2 / 3 - x**4 / 45
        else:
            exact_factor = x / np.tanh(x)

        optimal_feedback_at_q = z / T * exact_factor

        return [
            s * f * z,
            -T * (eta * v**2 + rho * q**2),
            -T * eta * (v - optimal_feedback_at_q)**2,
        ]

    def integrate(rtol, atol):
        solution = solve_ivp(
            dynamics,
            (1.0, 0.0),
            [q0, 0.0, 0.0],
            rtol=rtol,
            atol=atol,
        )
        assert solution.success, solution.message
        assert np.all(np.isfinite(solution.y))
        return solution.y[1, -1], solution.y[2, -1]

    cost, gap_integral = integrate(1e-10, 1e-11)
    refined_cost, refined_gap = integrate(1e-12, 1e-13)

    integration_change = abs(cost - refined_cost)
    identity_error = abs(
        (refined_cost - exact_cost) - refined_gap
    )

    assert integration_change < 1e-5, "Integration insuffisamment stable"
    assert identity_error < 1e-5, "Identite d'optimalite non verifiee"
    assert refined_gap >= -1e-12

    row = {
        "seed": seed,
        "training_seconds": metrics["training_seconds"],
        "max_inventory_error_shares": metrics["max_inventory_error_shares"],
        "max_relative_coefficient_error": metrics[
            "max_relative_value_coefficient_error"
        ],
        "predicted_value_eur": metrics["predicted_initial_value_eur"],
        "integrated_policy_cost_eur": refined_cost,
        "exact_optimal_cost_eur": exact_cost,
        "policy_gap_via_nonnegative_integral_eur": refined_gap,
        "integration_refinement_change_eur": integration_change,
        "optimality_identity_error_eur": identity_error,
    }
    rows.append(row)

    print(
        f"Erreur inventaire : {row['max_inventory_error_shares']:.6f} actions\n"
        f"Valeur predite : {row['predicted_value_eur']:.8f} EUR\n"
        f"Cout integre : {refined_cost:.8f} EUR\n"
        f"Cout optimal : {exact_cost:.8f} EUR\n"
        f"Ecart a l'optimum (integrale positive) : {refined_gap:.3e} EUR\n"
        f"Erreur identite : {identity_error:.3e} EUR",
        flush=True,
    )

output = root / "results" / "pinn_validation.csv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

labels = [str(row["seed"]) for row in rows]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(labels, [r["max_inventory_error_shares"] for r in rows])
axes[0].set(
    xlabel="Random seed",
    ylabel="Maximum sampled inventory error (shares)",
    title="Initialization sensitivity",
)

axes[1].bar(
    labels,
    [r["policy_gap_via_nonnegative_integral_eur"] for r in rows],
)
axes[1].set(
    xlabel="Random seed",
    ylabel="Policy cost minus optimum (EUR)",
    title="Integrated policy optimality gap",
)
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

for ax in axes:
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
figure = root / "figures" / "pinn_validation.png"
fig.savefig(figure, dpi=160)
plt.close(fig)

print("\nVerifications numeriques terminees.")
print("Trois graines, un seul jeu de parametres.")
print(f"Resultats : {output}")
print(f"Graphique : {figure}")
