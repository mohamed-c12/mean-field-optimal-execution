import numpy as np
from mean_field import best_response, q0, n_steps

risk_levels = np.array([0.0001, 0.001, 0.01])
weights = np.array([1 / 3, 1 / 3, 1 / 3])

mean_trades = np.full(n_steps, q0 / n_steps)
damping = 0.2
tolerance = 0.01
max_iterations = 300

for iteration in range(1, max_iterations + 1):
    responses = np.array([
        best_response(mean_trades, trader_risk_aversion=lam)
        for lam in risk_levels
    ])
    aggregate_response = weights @ responses
    residual = np.max(np.abs(aggregate_response - mean_trades))

    if iteration == 1 or iteration % 20 == 0:
        print(
            f"Iteration {iteration} | "
            f"ecart collectif : {residual:.6f} actions"
        )

    if residual < tolerance:
        break

    mean_trades = (
        (1.0 - damping) * mean_trades
        + damping * aggregate_response
    )
else:
    raise RuntimeError("Convergence non atteinte.")

print(f"\nEquilibre approche en {iteration} iterations")
print(f"Ecart collectif maximal : {residual:.6f} actions")

for lam, trades in zip(risk_levels, responses):
    remaining_at_30 = q0 - trades[:30].sum()
    print(
        f"lambda={lam:.4f} | "
        f"premiere vente={trades[0]:.2f} | "
        f"inventaire a 30 min={remaining_at_30:.2f} | "
        f"total vendu={trades.sum():.6f}"
    )

from pathlib import Path
import matplotlib.pyplot as plt
from mean_field import dt

times = np.arange(n_steps + 1) * dt
inventories = q0 - np.column_stack((
    np.zeros(len(risk_levels)),
    np.cumsum(responses, axis=1),
))
average_inventory = weights @ inventories

fig, ax = plt.subplots(figsize=(8, 5))

for lam, inventory in zip(risk_levels, inventories):
    ax.step(
        times, inventory, where="post",
        label=f"Risk aversion = {lam:g}"
    )

ax.step(
    times, average_inventory, where="post",
    color="black", linestyle="--", label="Population average"
)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Remaining inventory (shares)")
ax.set_title("Approximate heterogeneous mean-field equilibrium")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent
    / "figures" / "heterogeneous_inventory.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)

print(f"Graphique enregistre : {output_path}")

from mean_field import eta, sigma, gamma

# Flux effectivement produit par les strategies des trois groupes
realized_mean_trades = weights @ responses
fixed_decline = gamma * np.cumsum(realized_mean_trades)

def score(trades, lam):
    inventory_before = q0 - np.concatenate(
        ([0.0], np.cumsum(trades[:-1]))
    )
    return (
        eta / dt * np.sum(trades**2)
        + np.dot(trades, fixed_decline)
        + lam * sigma**2 * dt * np.sum(inventory_before**2)
    )

print("\nVerification des deviations individuelles :")

for lam, current_trades in zip(risk_levels, responses):
    deviation = best_response(
        realized_mean_trades, trader_risk_aversion=lam
    )
    current_score = score(current_trades, lam)
    best_score = score(deviation, lam)
    gain = current_score - best_score

    print(
        f"lambda={lam:.4f} | "
        f"score actuel={current_score:.8f} EUR | "
        f"gain par deviation={gain:.8f} EUR"
    )
