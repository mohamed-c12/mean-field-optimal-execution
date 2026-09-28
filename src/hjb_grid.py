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
for _ in range(n_time):
    total_cost = interval_cost + value[None, :]
    value = np.min(total_cost, axis=1)

kappa = np.sqrt(rho / eta)
exact_value = eta * kappa / np.tanh(kappa * T) * q0**2
relative_error = abs(value[-1] - exact_value) / exact_value

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
