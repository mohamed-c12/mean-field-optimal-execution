from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import expit

T = 60.0
ETA = 0.006
RHO = 0.001 * 0.20**2
GAMMA = 0.001
WEIGHTS = np.array([0.25, 0.50, 0.25])
INITIAL = np.array([500.0, 750.0, 1000.0])


def system_matrix(active):
    a = np.asarray(active, dtype=float)
    return np.block([
        [np.zeros((3, 3)), np.diag(a)],
        [
            (RHO / ETA) * np.diag(a),
            -GAMMA / (2 * ETA) * np.outer(a, WEIGHTS * a),
        ],
    ])


ALL_ACTIVE = system_matrix([1, 1, 1])
AFTER_EXIT = system_matrix([0, 1, 1])


def boundary_states(parameters):
    exit_time = T * expit(parameters[0])
    terminal = np.zeros(6)
    terminal[4:] = -np.exp(parameters[1:])
    at_exit = expm(AFTER_EXIT * (exit_time - T)) @ terminal
    at_start = expm(ALL_ACTIVE * (-exit_time)) @ at_exit
    return exit_time, terminal, at_exit, at_start


def boundary_error(parameters):
    return (
        boundary_states(parameters)[-1][:3] - INITIAL
    ) / 1000.0


fit = least_squares(
    boundary_error,
    x0=[-0.3, -4.0, -1.0],
    xtol=1e-12,
    ftol=1e-12,
    gtol=1e-12,
    max_nfev=200,
)

exit_time, terminal, at_exit, at_start = boundary_states(fit.x)
assert fit.success, fit.message
assert np.max(np.abs(at_start[:3] - INITIAL)) < 1e-6


def state(time):
    if time < exit_time:
        return expm(ALL_ACTIVE * (time - exit_time)) @ at_exit
    return expm(AFTER_EXIT * (time - T)) @ terminal


times = np.unique(np.append(np.linspace(0, T, 1201), exit_time))
states = np.array([state(t) for t in times])
inventories = states[:, :3]
rates = -states[:, 3:]
mean_inventory = inventories @ WEIGHTS

assert inventories.min() >= -1e-7
assert rates.min() >= -1e-7
assert np.max(np.diff(inventories, axis=0)) < 1e-7
assert np.max(np.abs(inventories[-1])) < 1e-7
assert abs(at_exit[0]) < 1e-8
assert abs(at_exit[3]) < 1e-8

# This active-set pattern is validated for the parameters above:
# group 1 exits early; groups 2 and 3 remain active until T.
# It is not a general solver for arbitrary parameter choices.
assert np.all(inventories[:-1, 1:] > 0)


def population_cost_rate(time):
    y = state(time)
    q = y[:3]
    v = -y[3:]
    mean_rate = WEIGHTS @ v
    return WEIGHTS @ (
        ETA * v**2 + RHO * q**2 + GAMMA * mean_rate * q
    )


continuous_score = sum(
    quad(population_cost_rate, left, right, epsabs=1e-8)[0]
    for left, right in [(0.0, exit_time), (exit_time, T)]
)
mean_at_30 = WEIGHTS @ state(30.0)[:3]

print(f"Liquidation du groupe 500 : {exit_time:.6f} minutes")
print(f"Inventaire moyen continu a 30 min : {mean_at_30:.6f}")
print(f"Score moyen continu : {continuous_score:.6f} EUR")
print(
    "Erreur sur les inventaires initiaux : "
    f"{np.max(np.abs(at_start[:3] - INITIAL)):.2e}"
)

root = Path(__file__).resolve().parents[1]
csv_path = root / "results" / "coupled_refinement.csv"

print("\nComparaison des inventaires moyens a 30 min :")
with csv_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        grid_inventory = float(row["mean_inventory_30min"])
        error = abs(grid_inventory - mean_at_30)
        print(
            f"dq={float(row['dq']):.2f} | "
            f"grille={grid_inventory:.6f} | "
            f"erreur absolue={error:.6f} actions"
        )

(root / "results").mkdir(exist_ok=True)
np.savetxt(
    root / "results" / "continuous_mean_field.csv",
    np.column_stack([times, inventories, mean_inventory]),
    delimiter=",",
    header="time,q_500,q_750,q_1000,mean_inventory",
    comments="",
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for i, q0 in enumerate(INITIAL):
    axes[0].plot(times, inventories[:, i], label=f"Initial: {q0:.0f}")
axes[0].set(
    title="Continuous equilibrium by initial inventory",
    xlabel="Time (minutes)",
    ylabel="Inventory (shares)",
)
axes[0].legend()

axes[1].plot(times, mean_inventory, label="Continuous mean inventory")
axes[1].axvline(
    exit_time, color="gray", linestyle="--",
    label="Exit of the 500-share group",
)
axes[1].set(
    title="Population mean inventory",
    xlabel="Time (minutes)",
    ylabel="Mean inventory (shares)",
)
axes[1].legend()

for ax in axes:
    ax.grid(alpha=0.3)

fig.tight_layout()
(root / "figures").mkdir(exist_ok=True)
figure = root / "figures" / "continuous_mean_field.png"
fig.savefig(figure, dpi=160)
plt.close(fig)

print("\nToutes les verifications ont reussi.")
print(f"Graphique : {figure}")
