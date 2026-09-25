import numpy as np

# Parametres illustratifs
initial_price = 100.0  # Euros par action
sigma = 0.20          # Euros par racine de minute
dt = 1.0              # Une minute par intervalle
n_steps = 60

# Variations independantes, puis prix cumules
rng = np.random.default_rng(42)
changes = sigma * np.sqrt(dt) * rng.standard_normal(n_steps)
prices = np.concatenate(([initial_price], initial_price + np.cumsum(changes)))

print(f"Prix initial : {prices[0]:.2f} euros")
print(f"Prix apres 30 minutes : {prices[30]:.2f} euros")
print(f"Prix apres 60 minutes : {prices[-1]:.2f} euros")

from pathlib import Path
import matplotlib.pyplot as plt

times = np.arange(n_steps + 1) * dt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(times, prices, label="Simulated market price")
ax.axhline(initial_price, color="gray", linestyle="--", label="Initial price")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Price (EUR)")
ax.set_title("Simulated price path over 60 minutes")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

output_path = Path(__file__).resolve().parent.parent / "figures" / "price_path.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)

print(f"Graphique enregistre : {output_path}")
