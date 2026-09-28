from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

T = 60.0
ETA = 0.006
RHO = 0.001 * 0.20**2
GAMMA = 0.001
SCALE = 1000.0
WEIGHTS = torch.tensor([0.25, 0.50, 0.25])
INITIAL = torch.tensor([0.50, 0.75, 1.00])
K2 = RHO * T**2 / ETA
COUPLING = GAMMA * T / (2 * ETA)


def network():
    return nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )


class CoupledTrajectories(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleList([network() for _ in range(3)])
        self.raw_exit = nn.Parameter(torch.tensor(0.0))

    def exit_fraction(self):
        return torch.sigmoid(self.raw_exit)

    def forward(self, x):
        u = x / self.exit_fraction()
        # Clamp the network input after exit; inventory is then zero.
        uc = torch.clamp(u, max=1.0)
        first = (
            INITIAL[0] * torch.relu(1 - u)**2
            * torch.exp(uc * self.nets[0](uc))
        )
        others = [
            INITIAL[i] * (1 - x) * torch.exp(x * self.nets[i](x))
            for i in (1, 2)
        ]
        return torch.cat([first] + others, dim=1)


model = CoupledTrajectories()


def derivatives(x):
    y = model(x)
    first = []
    second = []
    for i in range(3):
        dy = torch.autograd.grad(
            y[:, i].sum(), x, create_graph=True
        )[0]
        ddy = torch.autograd.grad(
            dy.sum(), x, create_graph=True
        )[0]
        first.append(dy)
        second.append(ddy)
    return y, torch.cat(first, 1), torch.cat(second, 1)


def equations(x):
    y, dy, ddy = derivatives(x)
    mean_derivative = (dy @ WEIGHTS).reshape(-1, 1)
    r = (ddy - K2 * y + COUPLING * mean_derivative) / (1 + K2)
    return r, dy


# Collocation points on each side of the learned exit.
u = ((torch.arange(128) + 0.5) / 128).reshape(-1, 1)


def loss_function():
    a = model.exit_fraction()
    before = a * u
    after = a + (1 - a) * u

    r_before, dy_before = equations(before)
    r_after, dy_after = equations(after)

    # The first group is inactive after its liquidation.
    physics = (
        r_before.square().mean()
        + r_after[:, 1:].square().mean()
    )
    selling = (
        torch.relu(dy_before).square().mean()
        + torch.relu(dy_after).square().mean()
    )
    return physics + 10 * selling


history = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
for step in range(5000):
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if (step + 1) % 500 == 0:
        print(
            f"Adam {step + 1}/5000 | loss={loss.item():.3e} | "
            f"sortie={T * model.exit_fraction().item():.4f} min",
            flush=True,
        )

print("Affinage L-BFGS...", flush=True)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    max_iter=800,
    tolerance_grad=1e-10,
    tolerance_change=1e-13,
    line_search_fn="strong_wolfe",
)


def closure():
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()
    return loss


optimizer.step(closure)

# Reference is loaded only after training.
root = Path(__file__).resolve().parents[1]
reference = np.loadtxt(
    root / "results" / "continuous_mean_field.csv",
    delimiter=",", skiprows=1,
)

times = np.linspace(0, T, 1201)
x = torch.tensor((times / T)[:, None], requires_grad=True)
y, dy, _ = derivatives(x)
q = SCALE * y.detach().numpy()
v = -SCALE / T * dy.detach().numpy()
weights = WEIGHTS.numpy()

q_reference = np.column_stack([
    np.interp(times, reference[:, 0], reference[:, i + 1])
    for i in range(3)
])
mean_q = q @ weights
mean_reference = q_reference @ weights
exit_time = T * model.exit_fraction().item()

# Separate residual check on shifted points, excluding the exit itself.
test_x = torch.tensor(
    ((np.arange(2000) + 0.37) / 2000)[:, None],
    requires_grad=True,
)
r, _ = equations(test_x)
r = r.detach().numpy()
active_first = test_x.detach().numpy().ravel() < exit_time / T
active_residuals = np.concatenate([
    r[active_first, 0], r[:, 1], r[:, 2]
])

assert np.all(np.isfinite(q))
assert np.all(np.isfinite(v))
assert np.max(np.abs(q[0] - [500, 750, 1000])) < 1e-8
assert np.max(np.abs(q[-1])) < 1e-8
assert q.min() >= -1e-8

metrics = {
    "seed": 42,
    "learned_first_exit_min": exit_time,
    "max_mean_inventory_error_shares": float(
        np.max(np.abs(mean_q - mean_reference))
    ),
    "max_group_inventory_errors_shares": (
        np.max(np.abs(q - q_reference), axis=0).tolist()
    ),
    "minimum_sampled_selling_rate": float(v.min()),
    "normalized_active_residual_rmse": float(
        np.sqrt(np.mean(active_residuals**2))
    ),
}

print("\nEvaluation du couplage :")
for key, value in metrics.items():
    print(f"{key} : {value}")

(root / "results").mkdir(exist_ok=True)
(root / "figures").mkdir(exist_ok=True)
(root / "results" / "pinn_coupled.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
)
np.savetxt(
    root / "results" / "pinn_coupled_paths.csv",
    np.column_stack([times, q, v, mean_q]),
    delimiter=",",
    header="time,q500,q750,q1000,v500,v750,v1000,mean_inventory",
    comments="",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].semilogy(np.arange(1, len(history) + 1), history)
axes[0].set(
    title="Adam training", xlabel="Iteration", ylabel="Loss",
)

axes[1].plot(times, mean_reference, "--", color="black", label="Reference")
axes[1].plot(times, mean_q, label="Coupled trajectory PINN")
axes[1].set(
    title="Population mean inventory",
    xlabel="Time (minutes)", ylabel="Shares",
)
axes[1].legend()

for i, initial in enumerate((500, 750, 1000)):
    axes[2].plot(
        times, q[:, i] - q_reference[:, i], label=str(initial)
    )
axes[2].set(
    title="Inventory errors by group",
    xlabel="Time (minutes)", ylabel="PINN minus reference (shares)",
)
axes[2].legend()

for ax in axes:
    ax.grid(alpha=0.3)

fig.tight_layout()
figure = root / "figures" / "pinn_coupled.png"
fig.savefig(figure, dpi=160)
plt.close(fig)
print(f"\nGraphique : {figure}")

if v.min() < -1e-6:
    print("ATTENTION : achats detectes, solution a corriger.")
