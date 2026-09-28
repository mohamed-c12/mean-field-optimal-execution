import numpy as np


def solve_population(mean_rate, n_inventory=400, gamma=0.001):
    """Bellman + transport pour un flux moyen impose, en actions/minute."""
    T = 60.0
    q_max = 1000.0
    eta = 0.006
    rho = 0.001 * 0.20**2

    mean_rate = np.asarray(mean_rate, dtype=float)
    if mean_rate.ndim != 1 or mean_rate.size == 0:
        raise ValueError("Le flux doit etre un vecteur non vide.")
    if not np.all(np.isfinite(mean_rate)) or np.any(mean_rate < 0):
        raise ValueError("Le flux doit etre fini et positif ou nul.")

    n_time = mean_rate.size
    dt = T / n_time
    grid = np.linspace(0.0, q_max, n_inventory + 1)
    before = grid[:, None]
    after = grid[None, :]

    base_cost = (
        eta * (before - after)**2 / dt
        + rho * dt / 3.0
        * (before**2 + before * after + after**2)
    )
    base_cost[after > before] = np.inf

    value = np.full(grid.size, np.inf)
    value[0] = 0.0
    policies = np.empty((n_time, grid.size), dtype=int)

    # Bellman en arriere : le flux est constant sur chaque intervalle.
    for k in range(n_time - 1, -1, -1):
        crowd_cost = gamma * mean_rate[k] * dt * (before + after) / 2.0
        candidates = base_cost + crowd_cost + value[None, :]
        policies[k] = np.argmin(candidates, axis=1)
        value = np.min(candidates, axis=1)

    # Population initiale : masses, et non densites.
    mass = np.zeros((n_time + 1, grid.size))
    for initial_q, weight in [(500.0, 0.25), (750.0, 0.50), (1000.0, 0.25)]:
        index = int(np.argmin(np.abs(grid - initial_q)))
        if not np.isclose(grid[index], initial_q):
            raise ValueError("La grille doit contenir les inventaires initiaux.")
        mass[0, index] += weight

    initial_value = mass[0] @ value
    forward_cost = 0.0

    # Transport en avant et calcul independant du cout des politiques.
    for k in range(n_time):
        destinations = policies[k]
        next_inventory = grid[destinations]
        trades = grid - next_inventory

        stage_cost = (
            eta * trades**2 / dt
            + rho * dt / 3.0
            * (grid**2 + grid * next_inventory + next_inventory**2)
            + gamma * mean_rate[k] * dt
            * (grid + next_inventory) / 2.0
        )
        forward_cost += mass[k] @ stage_cost
        np.add.at(mass[k + 1], destinations, mass[k])

    mean_inventory = mass @ grid
    output_rate = -np.diff(mean_inventory) / dt

    return {
        "times": np.linspace(0.0, T, n_time + 1),
        "mean_inventory": mean_inventory,
        "output_rate": output_rate,
        "value": initial_value,
        "forward_cost": forward_cost,
        "mass_error": np.max(np.abs(mass.sum(axis=1) - 1.0)),
        "minimum_mass": mass.min(),
        "liquidated_mass": mass[-1, 0],
    }
