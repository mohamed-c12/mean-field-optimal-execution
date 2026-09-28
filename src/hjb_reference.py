import numpy as np

q0 = 1000.0
T = 60.0
eta = 0.006
sigma = 0.20
risk_aversion = 0.001

kappa = np.sqrt(risk_aversion * sigma**2 / eta)

def value_coefficient(t):
    if not 0.0 <= t < T:
        raise ValueError("Il faut 0 <= t < T.")
    return eta * kappa / np.tanh(kappa * (T - t))

# Inventaire optimal en temps continu
def optimal_inventory(t):
    return q0 * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

initial_value = value_coefficient(0.0) * q0**2
initial_rate = value_coefficient(0.0) * q0 / eta

print(f"Cout optimal continu : {initial_value:.6f} EUR")
print(f"Vitesse initiale : {initial_rate:.4f} actions/minute")
print(f"Inventaire a 30 min : {optimal_inventory(30.0):.4f} actions")
print(f"Inventaire a 60 min : {optimal_inventory(T):.4f} actions")

print("\nConvergence du modele discret vers le continu :")

for n_steps in [60, 120, 240, 600, 1200]:
    dt = T / n_steps

    # Formule equivalente a arccosh, plus stable pour les petits pas
    theta = 2.0 * np.arcsinh(kappa * dt / 2.0)
    grid = np.arange(n_steps + 1)

    inventory = q0 * (
        np.sinh((n_steps - grid) * theta)
        / np.sinh(n_steps * theta)
    )
    trades = -np.diff(inventory)

    impact_cost = eta / dt * np.sum(trades**2)
    risk_penalty = (
        risk_aversion * sigma**2 * dt
        * np.sum(inventory[:-1]**2)
    )
    discrete_value = impact_cost + risk_penalty
    relative_error = abs(discrete_value - initial_value) / initial_value

    print(
        f"dt={dt:.3f} min | "
        f"cout={discrete_value:.6f} EUR | "
        f"ecart relatif={relative_error:.4%}"
    )
