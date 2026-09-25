import numpy as np
from scipy.optimize import minimize

initial_inventory = 1000.0
n_steps = 60
dt = 1.0
eta = 0.006
sigma = 0.20
risk_aversion = 0.001

def objective(trades):
    inventory_before = initial_inventory - np.concatenate(
        ([0.0], np.cumsum(trades[:-1]))
    )
    impact_cost = eta / dt * np.sum(trades**2)
    cost_variance = sigma**2 * dt * np.sum(inventory_before**2)
    return impact_cost + risk_aversion * cost_variance

# Point de depart : ventes egales, comme TWAP
twap_trades = np.full(n_steps, initial_inventory / n_steps)

# Vendre exactement 1000 actions au total
constraint = {
    "type": "eq",
    "fun": lambda trades: np.sum(trades) - initial_inventory,
}

# Chaque quantite vendue doit etre positive ou nulle
result = minimize(
    objective,
    x0=twap_trades,
    method="SLSQP",
    bounds=[(0.0, initial_inventory)] * n_steps,
    constraints=[constraint],
    options={"ftol": 1e-9, "maxiter": 1000},
)

if not result.success:
    raise RuntimeError(f"Echec de l'optimisation : {result.message}")

optimal_trades = result.x

print(f"Statut : {result.message}")
print(f"Quantite totale vendue : {optimal_trades.sum():.6f} actions")
print(f"Score theorique TWAP : {objective(twap_trades):.2f} euros")
print(f"Score optimise : {objective(optimal_trades):.2f} euros")
print("Cinq premieres ventes :", np.round(optimal_trades[:5], 2))
print("Cinq dernieres ventes :", np.round(optimal_trades[-5:], 2))

from pathlib import Path
import matplotlib.pyplot as plt

times = np.arange(n_steps + 1) * dt
twap_inventory = initial_inventory - np.concatenate(
    ([0.0], np.cumsum(twap_trades))
)
optimal_inventory = initial_inventory - np.concatenate(
    ([0.0], np.cumsum(optimal_trades))
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.step(times, twap_inventory, where="post", label="TWAP")
ax.step(times, optimal_inventory, where="post", label="Optimized execution")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Remaining inventory (shares)")
ax.set_title("TWAP vs optimized execution")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

output_path = Path(__file__).resolve().parent.parent / "figures" / "optimal_inventory.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)

print(f"Graphique enregistre : {output_path}")

# Solution analytique du modele discret
theta = np.arccosh(
    1.0 + risk_aversion * sigma**2 * dt**2 / (2.0 * eta)
)
grid = np.arange(n_steps + 1)
reference_inventory = initial_inventory * (
    np.sinh((n_steps - grid) * theta) / np.sinh(n_steps * theta)
)
reference_trades = -np.diff(reference_inventory)

max_error = np.max(np.abs(optimal_trades - reference_trades))

print(f"Ecart maximal par vente : {max_error:.6f} actions")
print(f"Score de reference : {objective(reference_trades):.6f} euros")
print(f"Score du solveur : {objective(optimal_trades):.6f} euros")
