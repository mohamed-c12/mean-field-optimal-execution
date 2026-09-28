import numpy as np

initial_inventory = 1000.0
n_steps = 60
dt = 1.0
eta = 0.006
sigma = 0.20
grid = np.arange(n_steps + 1)

for lam in [0.0, 0.0001, 0.001, 0.01]:
    if lam == 0.0:
        inventory = initial_inventory * (1.0 - grid / n_steps)
    else:
        theta = np.arccosh(
            1.0 + lam * sigma**2 * dt**2 / (2.0 * eta)
        )
        inventory = initial_inventory * (
            np.sinh((n_steps - grid) * theta)
            / np.sinh(n_steps * theta)
        )

    trades = -np.diff(inventory)
    expected_cost = eta / dt * np.sum(trades**2)
    std_cost = sigma * np.sqrt(dt * np.sum(inventory[:-1]**2))

    print(
        f"lambda={lam:.4f} | premiere vente={trades[0]:.2f} actions | "
        f"cout moyen={expected_cost:.2f} EUR | "
        f"ecart-type={std_cost:.2f} EUR"
    )
