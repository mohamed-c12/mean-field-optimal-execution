from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bellman_transport import solve_population

N_TIME = 120
N_INVENTORY = 200
GAMMA = 0.001
DT = 60.0 / N_TIME
TOLERANCE = 0.10  # Gain moyen maximal accepte, en EUR
MAX_ITERATIONS = 300


def evaluate(flow):
    result = solve_population(
        flow, n_inventory=N_INVENTORY, gamma=GAMMA
    )
    assert result["mass_error"] < 1e-10
    assert result["minimum_mass"] >= -1e-12
    assert abs(result["liquidated_mass"] - 1.0) < 1e-10
    assert abs(result["value"] - result["forward_cost"]) < 1e-6
    return result


def crowd_cost(flow, inventory):
    midpoint_inventory = (inventory[:-1] + inventory[1:]) / 2.0
    return GAMMA * DT * np.dot(flow, midpoint_inventory)


# Initialisation avec les strategies sans interaction
initial = evaluate(np.zeros(N_TIME))
mean_inventory = initial["mean_inventory"].copy()
mean_rate = initial["output_rate"].copy()
base_cost = initial["forward_cost"]
history = []

for iteration in range(1, MAX_ITERATIONS + 1):
    # Chaque trader prend le flux collectif actuel comme donne.
    response = evaluate(mean_rate)

    # Cout moyen des calendriers actuels face a leur propre flux
    current_score = base_cost + crowd_cost(mean_rate, mean_inventory)

    # Gain moyen possible en passant aux meilleures reponses
    gap = current_score - response["value"]
    history.append(gap)

    if gap < -1e-6:
        raise RuntimeError("Gain negatif significatif : verifier les couts.")

    if iteration == 1 or iteration % 10 == 0:
        print(
            f"Iteration {iteration} | "
            f"gain moyen par deviation : {gap:.6f} EUR"
        )

    if gap <= TOLERANCE:
        break

    # Retirer le cout collectif pour conserver seulement impact + risque.
    response_base_cost = (
        response["forward_cost"]
        - crowd_cost(mean_rate, response["mean_inventory"])
    )

    # Melange de populations suivant des calendriers complets.
    # On moyenne les couts des calendriers, pas le carre du flux moyen.
    alpha = 2.0 / (iteration + 2.0)
    mean_inventory = (
        (1.0 - alpha) * mean_inventory
        + alpha * response["mean_inventory"]
    )
    mean_rate = (
        (1.0 - alpha) * mean_rate
        + alpha * response["output_rate"]
    )
    base_cost = (
        (1.0 - alpha) * base_cost
        + alpha * response_base_cost
    )
else:
    raise RuntimeError("Tolerance non atteinte : ne pas conclure a un equilibre.")

# Coherence entre inventaire, flux et liquidation
assert np.all(mean_rate >= -1e-10)
assert np.max(np.abs(mean_rate + np.diff(mean_inventory) / DT)) < 1e-8
assert abs(DT * mean_rate.sum() - 750.0) < 1e-8
assert abs(mean_inventory[-1]) < 1e-8

print("\nCouplage termine :")
print(f"Iterations : {iteration}")
print(f"Gain moyen par deviation : {gap:.6f} EUR")
print(f"Score moyen de la population : {current_score:.6f} EUR")
print(f"Inventaire moyen initial : {mean_inventory[0]:.4f} actions")
print(f"Inventaire moyen a 30 min : {mean_inventory[N_TIME // 2]:.4f} actions")
print(f"Inventaire moyen final : {mean_inventory[-1]:.6f} actions")
print(f"Quantite moyenne vendue : {DT * mean_rate.sum():.6f} actions")
print("Toutes les verifications ont reussi.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(
    initial["times"], initial["mean_inventory"],
    label="No interactions"
)
axes[0].plot(
    initial["times"], mean_inventory,
    label="Approximate coupled equilibrium"
)
axes[0].set_xlabel("Time (minutes)")
axes[0].set_ylabel("Mean inventory (shares)")
axes[0].set_title("Population inventory")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].semilogy(
    np.arange(1, len(history) + 1),
    np.maximum(history, 1e-12)
)
axes[1].axhline(
    TOLERANCE, color="gray", linestyle="--",
    label="Stopping tolerance"
)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Average unilateral improvement (EUR)")
axes[1].set_title("Equilibrium check on the grid")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
output_path = (
    Path(__file__).resolve().parent.parent
    / "figures" / "coupled_bellman.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)
print(f"Graphique : {output_path}")
