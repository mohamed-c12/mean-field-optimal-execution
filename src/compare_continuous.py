from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from coupled_refinement import equilibrium

root = Path(__file__).resolve().parents[1]
reference = np.loadtxt(
    root / "results" / "continuous_mean_field.csv",
    delimiter=",",
    skiprows=1,
)
assert reference.shape[1] == 5
assert np.all(np.isfinite(reference))
assert np.all(np.diff(reference[:, 0]) > 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(
    reference[:, 0], reference[:, -1],
    color="black", linestyle="--", label="Continuous reference",
)
rows = []

for nq in (200, 400, 800):
    print(f"Comparaison : dq={1000 / nq:.2f}...", flush=True)
    times, inventory, diagnostics = equilibrium(nq)
    exact = np.interp(times, reference[:, 0], reference[:, -1])
    error = inventory - exact

    maximum = np.max(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    peak_time = times[np.argmax(np.abs(error))]

    rows.append({
        "dq": 1000 / nq,
        "max_error_shares_at_time_nodes": maximum,
        "rmse_shares_at_time_nodes": rmse,
        "time_of_max_error_min": peak_time,
        "average_deviation_gain_eur": diagnostics["deviation_gain_eur"],
    })

    label = f"dq = {1000 / nq:.2f}"
    axes[0].plot(times, inventory, label=label)
    axes[1].plot(times, error, label=label)

    print(
        f"  erreur maximale={maximum:.4f} actions | "
        f"RMSE={rmse:.4f} actions | "
        f"maximum a t={peak_time:.1f} min",
        flush=True,
    )

axes[0].set(
    title="Grid solutions versus continuous reference",
    xlabel="Time (minutes)",
    ylabel="Mean inventory (shares)",
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set(
    title="Signed error at time-grid nodes",
    xlabel="Time (minutes)",
    ylabel="Grid minus reference (shares)",
)

for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)

fig.tight_layout()
figure = root / "figures" / "continuous_comparison.png"
fig.savefig(figure, dpi=160)
plt.close(fig)

result = root / "results" / "continuous_comparison.csv"
with result.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\nComparaison terminee.")
print("Erreurs mesurees aux noeuds temporels, toutes les 0.5 minute.")
print("Elles combinent discretisation et tolerance de resolution.")
print(f"Graphique : {figure}")
print(f"Resultats : {result}")
