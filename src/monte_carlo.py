import numpy as np

# Parametres illustratifs
initial_price = 100.0
initial_inventory = 1000.0
sigma = 0.20
eta = 0.006
dt = 1.0
n_steps = 60
n_paths = 10_000

# Chaque ligne correspond a une trajectoire de 60 minutes
rng = np.random.default_rng(42)
changes = sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
prices = initial_price + np.cumsum(changes, axis=1)

# Ventes TWAP a la fin de chaque minute
quantity_per_trade = initial_inventory / n_steps
trading_rate = quantity_per_trade / dt
execution_prices = prices - eta * trading_rate

revenues = quantity_per_trade * np.sum(execution_prices, axis=1)
shortfalls = initial_inventory * initial_price - revenues

mean_cost = np.mean(shortfalls)
std_cost = np.std(shortfalls, ddof=1)
standard_error = std_cost / np.sqrt(n_paths)

print(f"Nombre de trajectoires : {n_paths}")
print(f"Shortfall moyen : {mean_cost:.2f} euros")
print(f"Ecart-type des shortfalls : {std_cost:.2f} euros")
print(f"Erreur standard de la moyenne : {standard_error:.2f} euros")
print("Moyenne theorique : 100.00 euros")

# Ecart-type theorique pour des ventes en fin d'intervalle
theoretical_std = sigma * initial_inventory * np.sqrt(
    dt * (n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps)
)
relative_error = abs(std_cost - theoretical_std) / theoretical_std

print(f"Ecart-type theorique : {theoretical_std:.2f} euros")
print(f"Ecart relatif sur l'ecart-type : {relative_error:.2%}")
