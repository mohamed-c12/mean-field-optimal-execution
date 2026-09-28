import numpy as np

q0 = 1000.0
T = 60.0
eta = 0.006
sigma = 0.20
risk_aversion = 0.001
rho = risk_aversion * sigma**2

n_time = 120
n_inventory = 200
dt = T / n_time
q_grid = np.linspace(0.0, q0, n_inventory + 1)

# Lignes : inventaire avant vente ; colonnes : inventaire apres vente
q = q_grid[:, None]
r = q_grid[None, :]

interval_cost = (
    eta * (q - r)**2 / dt
    + rho * dt / 3.0 * (q**2 + q*r + r**2)
)
interval_cost[r > q] = np.inf  # Achats interdits

# Condition terminale : liquidation obligatoire
value = np.full(n_inventory + 1, np.inf)
value[0] = 0.0

# Recursion de Bellman, de l'echeance vers l'instant initial
policy_steps = []

for _ in range(n_time):
    total_cost = interval_cost + value[None, :]
    policy_steps.append(np.argmin(total_cost, axis=1))
    value = np.min(total_cost, axis=1)

# Suivre les decisions optimales depuis l'inventaire initial
inventory_indices = [n_inventory]
current_index = n_inventory

for policy in reversed(policy_steps):
    current_index = int(policy[current_index])
    inventory_indices.append(current_index)

inventory_path = q_grid[inventory_indices]
optimal_trades = inventory_path[:-1] - inventory_path[1:]

kappa = np.sqrt(rho / eta)
exact_value = eta * kappa / np.tanh(kappa * T) * q0**2
relative_error = abs(value[-1] - exact_value) / exact_value

print(f"Premiere vente optimale : {optimal_trades[0]:.2f} actions")
print(f"Ventes sur les 5 premiers intervalles : {optimal_trades[:5]}")
print(f"Inventaire apres 30 minutes : {inventory_path[n_time // 2]:.2f} actions")
print(f"Pas de temps : {dt:.2f} minute")
print(f"Pas d'inventaire : {q_grid[1] - q_grid[0]:.2f} actions")
print(f"Valeur sur grille : {value[-1]:.6f} EUR")
print(f"Valeur exacte continue : {exact_value:.6f} EUR")
print(f"Ecart relatif : {relative_error:.4%}")

print("\nRaffinement de la grille d'inventaire, dt fixe :")

for n_q in [200, 400, 800]:
    grid = np.linspace(0.0, q0, n_q + 1)
    before = grid[:, None]
    after = grid[None, :]

    costs = (
        eta * (before - after)**2 / dt
        + rho * dt / 3.0
        * (before**2 + before*after + after**2)
    )
    costs[after > before] = np.inf

    values = np.full(n_q + 1, np.inf)
    values[0] = 0.0

    for _ in range(n_time):
        values = np.min(costs + values[None, :], axis=1)

    error = abs(values[-1] - exact_value) / exact_value
    print(
        f"dq={q0 / n_q:.2f} actions | "
        f"valeur={values[-1]:.6f} EUR | "
        f"ecart={error:.4%}"
    )

from pathlib import Path
import matplotlib.pyplot as plt

times = np.linspace(0.0, T, n_time + 1)
exact_inventory = (
    q0 * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)
)

# Cout du calendrier reconstruit
reconstructed_cost = np.sum(
    eta * optimal_trades**2 / dt
    + rho * dt / 3.0 * (
        inventory_path[:-1]**2
        + inventory_path[:-1] * inventory_path[1:]
        + inventory_path[1:]**2
    )
)

print(f"\nInventaire final : {inventory_path[-1]:.6f} actions")
print(f"Total vendu : {optimal_trades.sum():.6f} actions")
print(f"Cout du calendrier reconstruit : {reconstructed_cost:.6f} EUR")
print(f"Ecart avec Bellman : {abs(reconstructed_cost - value[-1]):.10f} EUR")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(times, exact_inventory, label="Exact continuous solution")
ax.plot(
    times, inventory_path, "--",
    label="Bellman grid: dt=0.5 min, dq=5 shares"
)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Remaining inventory (shares)")
ax.set_title("Bellman policy vs analytical solution")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent
    / "figures" / "hjb_inventory.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)
print(f"Graphique enregistre : {output_path}")

# Distribution initiale : masses de probabilite sur la grille
mass = np.zeros(n_inventory + 1)

for initial_q, weight in [(500.0, 0.25), (750.0, 0.50), (1000.0, 0.25)]:
    index = int(np.argmin(np.abs(q_grid - initial_q)))
    mass[index] += weight

mass_history = [mass.copy()]

# Transport de la population selon la politique optimale
for policy in reversed(policy_steps):
    next_mass = np.zeros_like(mass)
    np.add.at(next_mass, policy, mass)
    mass = next_mass
    mass_history.append(mass.copy())

mass_history = np.array(mass_history)
total_mass = mass_history.sum(axis=1)
mean_inventory = mass_history @ q_grid

print("\nTransport de la population sans interaction :")
print(f"Inventaire moyen initial : {mean_inventory[0]:.2f} actions")
print(f"Inventaire moyen a 30 min : {mean_inventory[n_time // 2]:.2f} actions")
print(f"Inventaire moyen final : {mean_inventory[-1]:.6f} actions")
print(f"Erreur maximale de masse : {np.max(np.abs(total_mass - 1.0)):.12f}")
print(f"Masse minimale : {mass_history.min():.12f}")
print(f"Proportion liquidee a T : {mass_history[-1, 0]:.6f}")

print("\nPrecision du transport, pas de temps fixe :")
reference_mean = 750.0 * np.sinh(kappa * T / 2) / np.sinh(kappa * T)
print(f"Reference continue a 30 min : {reference_mean:.4f} actions")

for n_q in [200, 400, 800]:
    grid = np.linspace(0.0, q0, n_q + 1)
    before = grid[:, None]
    after = grid[None, :]

    costs = (
        eta * (before - after)**2 / dt
        + rho * dt / 3.0
        * (before**2 + before*after + after**2)
    )
    costs[after > before] = np.inf

    values = np.full(n_q + 1, np.inf)
    values[0] = 0.0
    policies = []

    for _ in range(n_time):
        candidates = costs + values[None, :]
        policies.append(np.argmin(candidates, axis=1))
        values = np.min(candidates, axis=1)

    population = np.zeros(n_q + 1)
    for initial_q, weight in [(500.0, 0.25), (750.0, 0.50), (1000.0, 0.25)]:
        index = int(np.argmin(np.abs(grid - initial_q)))
        population[index] += weight

    mass_error = abs(population.sum() - 1.0)

    for step, policy in enumerate(reversed(policies), start=1):
        next_population = np.zeros_like(population)
        np.add.at(next_population, policy, population)
        population = next_population

        mass_error = max(mass_error, abs(population.sum() - 1.0))

        if step == n_time // 2:
            mean_at_30 = population @ grid

    print(
        f"dq={q0 / n_q:.2f} | "
        f"inventaire moyen a 30 min={mean_at_30:.4f} | "
        f"erreur absolue={abs(mean_at_30 - reference_mean):.4f} actions | "
        f"erreur de masse={mass_error:.2e}"
    )
