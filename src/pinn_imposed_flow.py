from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
import torch
from torch import nn

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

T = 60.0
Q0 = 1000.0
ETA = 0.006
RHO = 0.001 * 0.20**2
GAMMA = 0.001
MEAN_RATE = 12.5
CROWD = GAMMA * MEAN_RATE


class Trajectory(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.raw_tau = nn.Parameter(torch.tensor(0.0))

    def liquidation_time(self):
        return T * torch.sigmoid(self.raw_tau)

    def forward(self, u):
        # u = t / tau.
        # y(0)=1, y(1)=0 and y'(1)=0 exactly.
        return (1 - u)**2 * torch.exp(u * self.net(u))


model = Trajectory()


def residual(points):
    u = points.detach().clone().requires_grad_(True)
    y = model(u)
    dy = torch.autograd.grad(
        y, u, torch.ones_like(y), create_graph=True
    )[0]
    ddy = torch.autograd.grad(
        dy, u, torch.ones_like(dy), create_graph=True
    )[0]
    tau = model.liquidation_time()
    raw = (
        ddy
        - tau**2 * RHO / ETA * y
        - tau**2 * CROWD / (2 * ETA * Q0)
    )
    return raw / (1 + tau**2 * RHO / ETA)


points = ((torch.arange(256) + 0.5) / 256).reshape(-1, 1)
history = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
for step in range(3000):
    optimizer.zero_grad()
    loss = residual(points).square().mean()
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if (step + 1) % 500 == 0:
        print(
            f"Adam {step + 1}/3000 | loss={loss.item():.3e} | "
            f"tau={model.liquidation_time().item():.4f} min",
            flush=True,
        )

print("Affinage L-BFGS...", flush=True)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    max_iter=500,
    tolerance_grad=1e-10,
    tolerance_change=1e-13,
    line_search_fn="strong_wolfe",
)


def closure():
    optimizer.zero_grad()
    loss = residual(points).square().mean()
    loss.backward()
    return loss


optimizer.step(closure)
tau = model.liquidation_time().item()

# Analytical reference is used only after training.
k = np.sqrt(RHO / ETA)
D = CROWD / (2 * RHO)
tau_exact = np.arccosh(1 + Q0 / D) / k
assert tau_exact < T, "This benchmark assumes early liquidation."


def learned_path(times):
    u = torch.tensor(
        np.clip(times / tau, 0, 1)[:, None],
        requires_grad=True,
    )
    y = model(u)
    dy = torch.autograd.grad(y, u, torch.ones_like(y))[0]
    q = Q0 * y.detach().numpy().ravel()
    v = -Q0 / tau * dy.detach().numpy().ravel()
    q[times >= tau] = 0
    v[times >= tau] = 0
    return q, v


def exact_path(times):
    remaining = np.maximum(tau_exact - times, 0)
    return (
        D * (np.cosh(k * remaining) - 1),
        D * k * np.sinh(k * remaining),
    )


def evaluate_cost(n):
    # Include both liquidation dates in the integration grid.
    times = np.unique(np.r_[
        np.linspace(0, T, n), tau, tau_exact
    ])
    q, v = learned_path(times)
    qe, ve = exact_path(times)
    cost = simpson(ETA * v**2 + RHO * q**2 + CROWD * q, x=times)
    exact_cost = simpson(
        ETA * ve**2 + RHO * qe**2 + CROWD * qe, x=times
    )
    return times, q, v, qe, cost, exact_cost


_, _, _, _, coarse_cost, _ = evaluate_cost(6001)
times, q, v, qe, cost, exact_cost = evaluate_cost(12001)

assert np.all(np.isfinite(q))
assert np.all(np.isfinite(v))
assert abs(q[0] - Q0) < 1e-8
assert abs(q[-1]) < 1e-8
assert q.min() >= -1e-8
assert v.min() >= -1e-7, "Buying detected: inspect the trained solution."
assert abs(cost - coarse_cost) < 1e-5
assert cost >= exact_cost - 1e-5

test = torch.linspace(0, 1, 1001).reshape(-1, 1)
test_r = residual(test).detach().numpy()

metrics = {
    "seed": 42,
    "imposed_mean_rate": MEAN_RATE,
    "learned_liquidation_time_min": tau,
    "exact_liquidation_time_min": tau_exact,
    "max_sampled_inventory_error_shares": float(np.max(np.abs(q - qe))),
    "normalized_test_residual_rmse": float(np.sqrt(np.mean(test_r**2))),
    "integrated_policy_cost_eur": float(cost),
    "exact_policy_cost_eur": float(exact_cost),
    "policy_cost_gap_eur": float(cost - exact_cost),
    "quadrature_refinement_change_eur": float(abs(cost - coarse_cost)),
}

print("\nEvaluation :")
for name, value in metrics.items():
    print(f"{name} : {value:.10g}")

root = Path(__file__).resolve().parents[1]
for folder in ("results", "figures"):
    (root / folder).mkdir(exist_ok=True)

(root / "results" / "pinn_imposed_flow.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].semilogy(np.arange(1, len(history) + 1), history)
axes[0].set(
    title="Adam training", xlabel="Iteration",
    ylabel="Normalized residual MSE",
)

axes[1].plot(times, qe, "--", color="black", label="Analytical")
axes[1].plot(times, q, label="PINN")
axes[1].axvline(tau_exact, color="gray", linestyle=":")
axes[1].set(
    title="Response to imposed mean flow",
    xlabel="Time (minutes)", ylabel="Inventory (shares)",
)
axes[1].legend()

axes[2].plot(times, q - qe)
axes[2].set(
    title="Inventory error",
    xlabel="Time (minutes)", ylabel="PINN minus analytical (shares)",
)

for ax in axes:
    ax.grid(alpha=0.3)

fig.tight_layout()
figure = root / "figures" / "pinn_imposed_flow.png"
fig.savefig(figure, dpi=160)
plt.close(fig)

print("\nVerifications numeriques terminees.")
print(f"Graphique : {figure}")
