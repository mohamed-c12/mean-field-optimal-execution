from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from bellman_transport import solve_population

N_TIME = 120
DT = 60.0 / N_TIME
GAMMA = 0.001
TOLERANCE = 0.10
MAX_ITERATIONS = 500


def crowd_cost(flow, inventory):
    midpoint = (inventory[:-1] + inventory[1:]) / 2
    return GAMMA * DT * np.dot(flow, midpoint)


def equilibrium(n_inventory):
    def solve(flow):
        result = solve_population(
            flow, n_inventory=n_inventory, gamma=GAMMA
        )
        assert result["mass_error"] < 1e-10
        assert result["minimum_mass"] >= -1e-12
        assert abs(result["liquidated_mass"] - 1.0) < 1e-10
        assert abs(result["value"] - result["forward_cost"]) < 1e-6
        return result

    initial = solve(np.zeros(N_TIME))
    inventory = initial["mean_inventory"].copy()
    flow = initial["output_rate"].copy()
    base_cost = initial["forward_cost"]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = solve(flow)
        score = base_cost + crowd_cost(flow, inventory)
        gap = score - response["value"]
        assert gap >= -1e-6

        if gap <= TOLERANCE:
            break

        if iteration == MAX_ITERATIONS:
            raise RuntimeError(
                f"Tolerance non atteinte pour nq={n_inventory}: {gap}"
            )

        response_base = (
            response["forward_cost"]
            - crowd_cost(flow, response["mean_inventory"])
        )
        alpha = 2.0 / (iteration + 2.0)
        inventory = (
            (1 - alpha) * inventory
            + alpha * response["mean_inventory"]
        )
        flow = (
            (1 - alpha) * flow
            + alpha * response["output_rate"]
        )
        base_cost = (1 - alpha) * base_cost + alpha * response_base

    assert np.all(flow >= -1e-10)
    assert np.all(np.diff(inventory) <= 1e-10)
    assert abs(inventory[0] - 750.0) < 1e-8
    assert abs(inventory[-1]) < 1e-8
    assert abs(DT * flow.sum() - 750.0) < 1e-8
    assert np.max(np.abs(flow + np.diff(inventory) / DT)) < 1e-8

    return initial["times"], inventory, {
        "dq": 1000.0 / n_inventory,
        "iterations": iteration,
        "deviation_gain_eur": gap,
        "mean_inventory_30min": inventory[N_TIME // 2],
        "population_score_eur": score,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    rows = []
    paths = []
    fig, ax = plt.subplots(figsize=(8, 5))

    for nq in (200, 400, 800):
        print(f"Calcul en cours : dq={1000 / nq:.2f} actions...", flush=True)
        times, inventory, row = equilibrium(nq)
        rows.append(row)
        paths.append(inventory)
        ax.plot(times, inventory, label=f"dq = {row['dq']:.2f} shares")
        print(
            f"  iterations={row['iterations']} | "
            f"gain={row['deviation_gain_eur']:.6f} EUR | "
            f"inventaire a 30 min={row['mean_inventory_30min']:.4f} | "
            f"score={row['population_score_eur']:.6f} EUR",
            flush=True,
        )

    print("\nEcarts maximaux entre trajectoires moyennes :")
    for i in range(1, len(paths)):
        difference = np.max(np.abs(paths[i] - paths[i - 1]))
        print(
            f"dq={rows[i-1]['dq']:.2f} -> {rows[i]['dq']:.2f} : "
            f"{difference:.4f} actions"
        )

    csv_path = root / "results" / "coupled_refinement.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ax.set(
        xlabel="Time (minutes)",
        ylabel="Mean inventory (shares)",
        title="Coupled equilibrium: inventory grid refinement",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    figure_path = root / "figures" / "coupled_refinement.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("\nToutes les verifications ont reussi.")
    print(f"Resultats : {csv_path}")
    print(f"Graphique : {figure_path}")


if __name__ == "__main__":
    main()
