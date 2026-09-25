from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 1. Paramètres du problème
initial_inventory = 1000.0  # Nombre initial d'actions
horizon_minutes = 60.0     # Durée totale, en minutes
n_trades = 60              # Nombre de ventes

# 2. Discrétisation du temps
dt = horizon_minutes / n_trades
times = np.linspace(0.0, horizon_minutes, n_trades + 1)

# 3. Stratégie TWAP : vitesse de vente constante
trading_rate = initial_inventory / horizon_minutes
quantity_per_trade = trading_rate * dt

# 4. Inventaire aux instants de la grille
inventory = initial_inventory - trading_rate * times

# 5. Affichage des résultats
print(f"Inventaire initial : {inventory[0]:.2f} actions")
print(f"Intervalle entre ventes : {dt:.2f} minute(s)")
print(f"Vitesse de vente : {trading_rate:.2f} actions/minute")
print(f"Quantité par vente : {quantity_per_trade:.2f} actions")
print(f"Inventaire après 30 minutes : {inventory[n_trades // 2]:.2f} actions")
print(f"Inventaire final : {inventory[-1]:.2f} actions")

# 6. Graphique de l'inventaire
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(times, inventory, color="navy", linewidth=2, label="TWAP")
ax.set_xlabel("Temps (minutes)")
ax.set_ylabel("Actions restant à vendre")
ax.set_title("Liquidation de 1 000 actions en 60 minutes")
ax.set_xlim(0, horizon_minutes)
ax.set_ylim(0, initial_inventory * 1.05)
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

project_root = Path(__file__).resolve().parent.parent
output_path = project_root / "figures" / "twap_inventory.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output_path, dpi=150)
plt.close(fig)

print(f"Graphique enregistré : {output_path}")

# 7. Cout de l'impact temporaire : parametres illustratifs
market_price = 100.0  # Euros par action, suppose constant
eta = 0.006          # Euros * minute / action^2

execution_price = market_price - eta * trading_rate
revenue = execution_price * initial_inventory
temporary_cost = eta * trading_rate * initial_inventory

print(f"Prix de vente par action : {execution_price:.2f} euros")
print(f"Recette totale : {revenue:.2f} euros")
print(f"Cout temporaire TWAP : {temporary_cost:.2f} euros")
