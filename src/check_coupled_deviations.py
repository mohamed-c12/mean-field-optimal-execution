from pathlib import Path
import csv

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
ETA = 0.006
RHO = 0.001 * 0.20**2
GAMMA = 0.001
WEIGHTS = np.array([0.25, 0.50, 0.25])
INITIAL = np.array([500.0, 750.0, 1000.0])
rows = []


def evaluate(z, q0, dt, mean_rate):
    # z contains sale quantities divided by initial inventory.
    sales = q0 * z
    before = q0 - np.r_[0.0, np.cumsum(sales)[:-1]]
    after = before - sales

    cost = np.sum(
        ETA * sales**2 / dt
        + RHO * dt / 3 * (before**2 + before * after + after**2)
        + GAMMA * mean_rate * dt * (before + after) / 2
    )

    # Analytical gradient, with the population flow held fixed.
    future_terms = (
        RHO * dt * (before + after) + GAMMA * mean_rate * dt
    )
    future_sum = np.cumsum(future_terms[::-1])[::-1] - future_terms
    gradient = (
        2 * ETA * sales / dt
        - RHO * dt / 3 * (before + 2 * after)
        - GAMMA * mean_rate * dt / 2
        - future_sum
    )
    return cost / 1000.0, gradient * q0 / 1000.0


for seed in (0, 1, 42):
    suffix = "" if seed == 42 else f"_seed_{seed}"
    data = np.loadtxt(
        ROOT / "results" / f"pinn_coupled_paths{suffix}.csv",
        delimiter=",", skiprows=1,
    )

    for n in (120, 240):
        dt = 60.0 / n
        times = np.linspace(0, 60, n + 1)
        inventory = np.column_stack([
            np.interp(times, data[:, 0], data[:, i + 1])
            for i in range(3)
        ])
        sales = -np.diff(inventory, axis=0)
        assert sales.min() >= -1e-8
        assert np.max(np.abs(sales.sum(axis=0) - INITIAL)) < 1e-6

        mean_rate = (sales @ WEIGHTS) / dt

        for group, q0 in enumerate(INITIAL):
            print(
                f"seed={seed} | dt={dt:.2f} | q0={q0:.0f}...",
                flush=True,
            )
            start = sales[:, group] / q0
            current_cost = 1000 * evaluate(
                start, q0, dt, mean_rate
            )[0]

            result = minimize(
                evaluate,
                start,
                args=(q0, dt, mean_rate),
                jac=True,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints={
                    "type": "eq",
                    "fun": lambda z: z.sum() - 1,
                    "jac": lambda z: np.ones_like(z),
                },
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            assert result.success, result.message
            assert abs(result.x.sum() - 1) < 1e-8
            assert result.x.min() >= -1e-10

            best_cost = 1000 * result.fun
            gain = current_cost - best_cost
            assert gain >= -1e-6, "Best response worsened the objective."

            rows.append({
                "seed": seed,
                "dt_min": dt,
                "initial_inventory": q0,
                "sampled_pinn_cost_eur": current_cost,
                "best_response_cost_eur": best_cost,
                "estimated_deviation_gain_eur": gain,
                "solver_iterations": result.nit,
            })

output = ROOT / "results" / "coupled_deviations.csv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\nBILAN DES DEVIATIONS")
for seed in (0, 1, 42):
    for dt in (0.5, 0.25):
        selected = [
            r for r in rows
            if r["seed"] == seed and r["dt_min"] == dt
        ]
        gains = np.array([
            r["estimated_deviation_gain_eur"] for r in selected
        ])
        print(
            f"seed={seed} | dt={dt:.2f} | "
            f"gain moyen={WEIGHTS @ gains:.8f} EUR | "
            f"gain maximal={gains.max():.8f} EUR"
        )

print(f"\nResultats : {output}")
print("Diagnostic discret : trajectoires lineaires par intervalle.")
