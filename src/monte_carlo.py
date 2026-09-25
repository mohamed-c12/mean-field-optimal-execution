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

# Strategie acceleree : liquidation en 30 minutes
# On utilise les memes trajectoires que pour TWAP.
fast_steps = 30
fast_quantity = initial_inventory / fast_steps
fast_rate = fast_quantity / dt

fast_execution_prices = prices[:, :fast_steps] - eta * fast_rate
fast_revenues = fast_quantity * np.sum(fast_execution_prices, axis=1)
fast_shortfalls = initial_inventory * initial_price - fast_revenues

print("\nComparaison sur les memes trajectoires :")
print(f"TWAP 60 min : moyenne = {mean_cost:.2f}, ecart-type = {std_cost:.2f} euros")
print(
    f"Vente 30 min : moyenne = {fast_shortfalls.mean():.2f}, "
    f"ecart-type = {fast_shortfalls.std(ddof=1):.2f} euros"
)

# Critere moyenne-variance : plus le score est faible, mieux c'est
risk_aversion = 0.001  # En inverse d'euros

twap_score = mean_cost + risk_aversion * np.var(shortfalls, ddof=1)
fast_score = fast_shortfalls.mean() + risk_aversion * np.var(
    fast_shortfalls, ddof=1
)

print(f"\nAversion au risque : {risk_aversion}")
print(f"Score TWAP 60 min : {twap_score:.2f} euros")
print(f"Score vente 30 min : {fast_score:.2f} euros")

# Strategie optimale determinee sans utiliser les trajectoires simulees
theta = np.arccosh(
    1.0 + risk_aversion * sigma**2 * dt**2 / (2.0 * eta)
)
grid = np.arange(n_steps + 1)
optimal_inventory = initial_inventory * (
    np.sinh((n_steps - grid) * theta) / np.sinh(n_steps * theta)
)
optimal_trades = -np.diff(optimal_inventory)

# Ventes en fin de minute, avec impact propre a chaque quantite
optimal_execution_prices = prices - eta * optimal_trades / dt
optimal_revenues = np.sum(
    optimal_execution_prices * optimal_trades, axis=1
)
optimal_shortfalls = initial_inventory * initial_price - optimal_revenues

optimal_mean = optimal_shortfalls.mean()
optimal_std = optimal_shortfalls.std(ddof=1)
optimal_score = optimal_mean + risk_aversion * optimal_std**2

print("\nStrategie optimale sur les memes trajectoires :")
print(f"Shortfall moyen : {optimal_mean:.2f} euros")
print(f"Ecart-type : {optimal_std:.2f} euros")
print(f"Score simule : {optimal_score:.2f} euros")
