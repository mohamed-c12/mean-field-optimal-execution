import numpy as np

initial_price = 100.0
horizon = 60.0
n_steps = 60
dt = horizon / n_steps

# Flux moyen par trader, impose pour cette premiere experience
mean_initial_inventory = 1000.0
mean_trading_rate = mean_initial_inventory / horizon

# Impact permanent du flux moyen : EUR / action
gamma = 0.001

times = np.arange(n_steps + 1) * dt
mean_cumulative_sales = mean_trading_rate * times

# Prix moyen theorique : le bruit brownien est d'esperance nulle
expected_prices = initial_price - gamma * mean_cumulative_sales

for k in [0, 30, 60]:
    print(
        f"Temps : {times[k]:.0f} min | "
        f"Ventes moyennes cumulees : {mean_cumulative_sales[k]:.2f} actions | "
        f"Prix moyen : {expected_prices[k]:.2f} EUR"
    )

from scipy.optimize import minimize

initial_inventory = 1000.0
eta = 0.006
sigma = 0.20
risk_aversion = 0.001

def objective(trades, price_decline):
    inventory_before = initial_inventory - np.concatenate(
        ([0.0], np.cumsum(trades[:-1]))
    )
    temporary_cost = eta / dt * np.sum(trades**2)
    crowd_cost = np.dot(trades, price_decline)
    variance = sigma**2 * dt * np.sum(inventory_before**2)
    return temporary_cost + crowd_cost + risk_aversion * variance

twap_trades = np.full(n_steps, initial_inventory / n_steps)
constraint = {
    "type": "eq",
    "fun": lambda trades: trades.sum() - initial_inventory,
}

scenarios = {
    "Sans pression vendeuse": np.zeros(n_steps),
    "Avec pression vendeuse": initial_price - expected_prices[1:],
}

for name, decline in scenarios.items():
    result = minimize(
        objective,
        x0=twap_trades,
        args=(decline,),
        method="SLSQP",
        bounds=[(0.0, initial_inventory)] * n_steps,
        constraints=[constraint],
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"{name} : {result.message}")

    trades = result.x
    remaining_at_30 = initial_inventory - trades[:30].sum()

    print(f"\n{name}")
    print(f"Premiere vente : {trades[0]:.2f} actions")
    print(f"Inventaire apres 30 min : {remaining_at_30:.2f} actions")
    print(f"Total vendu : {trades.sum():.6f} actions")
