import numpy as np
from scipy.optimize import minimize

q0 = 1000.0
n_steps = 60
dt = 1.0
eta = 0.006
sigma = 0.20
risk_aversion = 0.001
gamma = 0.001

def best_response(mean_trades, trader_risk_aversion=risk_aversion):
    # mean_trades : quantites moyennes vendues par intervalle
    # Convention : impact collectif inclus dans le prix de fin d'intervalle
    price_decline = gamma * np.cumsum(mean_trades)

    def objective(trades):
        inventory_before = q0 - np.concatenate(
            ([0.0], np.cumsum(trades[:-1]))
        )
        impact = eta / dt * np.sum(trades**2)
        crowd_cost = np.dot(trades, price_decline)
        variance = sigma**2 * dt * np.sum(inventory_before**2)
        return impact + crowd_cost + trader_risk_aversion * variance

    result = minimize(
        objective,
        x0=np.full(n_steps, q0 / n_steps),
        method="SLSQP",
        bounds=[(0.0, q0)] * n_steps,
        constraints=[{
            "type": "eq",
            "fun": lambda trades: trades.sum() - q0,
        }],
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)

    return result.x

if __name__ == "__main__":
    mean_trades = np.full(n_steps, q0 / n_steps)
    response = best_response(mean_trades)

    print(f"Premiere vente moyenne supposee : {mean_trades[0]:.2f}")
    print(f"Premiere vente en meilleure reponse : {response[0]:.2f}")
    print(f"Total vendu en meilleure reponse : {response.sum():.6f}")

    # Recherche d'un equilibre par meilleures reponses amorties
    damping = 0.2
    tolerance = 0.01  # Ecart maximal accepte, en actions par intervalle
    max_iterations = 300

    for iteration in range(1, max_iterations + 1):
        response = best_response(mean_trades)
        residual = np.max(np.abs(response - mean_trades))

        if iteration == 1 or iteration % 20 == 0:
            print(
                f"Iteration {iteration} | "
                f"ecart maximal : {residual:.6f} actions"
            )

        if residual < tolerance:
            break

        mean_trades = (
            (1.0 - damping) * mean_trades + damping * response
        )
    else:
        raise RuntimeError(
            "Convergence non atteinte : ne pas conclure a un equilibre."
        )

    inventory = q0 - np.concatenate(([0.0], np.cumsum(mean_trades)))

    print(f"\nEquilibre approche atteint en {iteration} iterations")
    print(f"Ecart maximal moyenne / meilleure reponse : {residual:.6f}")
    print(f"Premiere vente : {mean_trades[0]:.2f} actions")
    print(f"Inventaire apres 30 min : {inventory[30]:.2f} actions")
    print(f"Total vendu : {mean_trades.sum():.6f} actions")

    # Le flux collectif reste fixe pendant cette verification
    fixed_decline = gamma * np.cumsum(mean_trades)

    def individual_score(trades):
        inventory_before = q0 - np.concatenate(
            ([0.0], np.cumsum(trades[:-1]))
        )
        return (
            eta / dt * np.sum(trades**2)
            + np.dot(trades, fixed_decline)
            + risk_aversion * sigma**2 * dt
            * np.sum(inventory_before**2)
        )

    equilibrium_score = individual_score(mean_trades)
    response_score = individual_score(response)
    improvement = equilibrium_score - response_score

    print(f"Score du calendrier moyen : {equilibrium_score:.8f} EUR")
    print(f"Score de la meilleure reponse : {response_score:.8f} EUR")
    print(f"Gain estime par deviation individuelle : {improvement:.8f} EUR")

    from pathlib import Path
    import matplotlib.pyplot as plt

    times = np.arange(n_steps + 1) * dt

    # Sans flux collectif : meme risque et meme impact temporaire
    no_interaction_trades = best_response(np.zeros(n_steps))
    no_interaction_inventory = q0 - np.concatenate(
        ([0.0], np.cumsum(no_interaction_trades))
    )
    twap_inventory = q0 * (1.0 - times / (n_steps * dt))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(
        times, twap_inventory, where="post",
        linestyle="--", label="TWAP"
    )
    ax.step(
        times, no_interaction_inventory, where="post",
        label="Optimal execution without interactions"
    )
    ax.step(
        times, inventory, where="post",
        label="Approximate mean-field equilibrium"
    )

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Remaining inventory (shares)")
    ax.set_title("Effect of mean-field interactions on execution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = (
        Path(__file__).resolve().parent.parent
        / "figures" / "mean_field_inventory.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Graphique enregistre : {output_path}")
