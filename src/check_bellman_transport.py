from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bellman_transport import solve_population

n_time = 120
results = {}

scenarios = {
    "No interactions": np.zeros(n_time),
    "Imposed population TWAP": np.full(n_time, 750.0 / 60.0),
}

for name, mean_rate in scenarios.items():
    result = solve_population(mean_rate, n_inventory=400)
    results[name] = result

    gap = abs(result["value"] - result["forward_cost"])
    inventory = result["mean_inventory"]

    assert result["mass_error"] < 1e-10, "Masse non conservee"
    assert result["minimum_mass"] >= -1e-12, "Masse negative"
    assert abs(result["liquidated_mass"] - 1.0) < 1e-10
    assert gap < 1e-6, "Incoherence entre Bellman et le cout reconstruit"
    assert np.all(np.diff(inventory) <= 1e-8), "Inventaire croissant"

    print(f"\n{name}")
    print(f"Inventaire moyen initial : {inventory[0]:.4f} actions")
    print(f"Inventaire moyen a 30 min : {inventory[n_time // 2]:.4f} actions")
    print(f"Inventaire moyen final : {inventory[-1]:.6f} actions")
    print(f"Valeur moyenne de Bellman : {result['value']:.6f} EUR")
    print(f"Ecart avec le cout reconstruit : {gap:.10f} EUR")
    print(f"Erreur de masse : {result['mass_error']:.2e}")

fig, ax = plt.subplots(figsize=(8, 5))
for name, result in results.items():
    ax.plot(result["times"], result["mean_inventory"], label=name)

ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Mean remaining inventory (shares)")
ax.set_title("Population response to an imposed mean trading flow")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent
    / "figures" / "bellman_population_response.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)

print("\nToutes les verifications ont reussi.")
print(f"Graphique : {output_path}")
