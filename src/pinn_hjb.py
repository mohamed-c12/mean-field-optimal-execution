from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
import torch
from torch import nn

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

T = 60.0
Q0 = 1000.0
ETA = 0.006
RHO = 0.001 * 0.20**2
K2 = RHO * T**2 / ETA
K = np.sqrt(K2)

# Positive, smooth correction to the terminal singularity.
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
    nn.Softplus(),
)


def residual(points):
    s = points.detach().clone().requires_grad_(True)
    f = model(s)
    derivative = torch.autograd.grad(
        f, s, grad_outputs=torch.ones_like(f), create_graph=True
    )[0]
    return (3 * f + s * derivative + s**2 * f**2 - K2) / K2


# Midpoints for training; a different grid is used for evaluation.
train_s = ((torch.arange(256) + 0.5) / 256).reshape(-1, 1)
history = []
started = time.perf_counter()

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
for step in range(2000):
    optimizer.zero_grad()
    loss = residual(train_s).square().mean()
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if (step + 1) % 500 == 0:
        print(
            f"Adam {step + 1}/2000 | loss={loss.item():.3e}",
            flush=True,
        )

print("Affinage avec L-BFGS...", flush=True)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=400,
    tolerance_grad=1e-10,
    tolerance_change=1e-13,
    line_search_fn="strong_wolfe",
)


def closure():
    optimizer.zero_grad()
    loss = residual(train_s).square().mean()
    loss.backward()
    return loss


optimizer.step(closure)
training_seconds = time.perf_counter() - started

test_s = torch.linspace(0, 1, 1001).reshape(-1, 1)
test_residual = residual(test_s).detach().numpy().ravel()

# Evaluate a(t) away from its singular terminal point.
s_value = np.linspace(0.001, 1.0, 1001)
with torch.no_grad():
    f_value = model(torch.tensor(s_value[:, None])).numpy().ravel()

a_predicted = ETA / T * (1 / s_value + s_value * f_value)
a_exact = ETA / T * K / np.tanh(K * s_value)
relative_error = np.abs(a_predicted - a_exact) / a_exact

# Recover the inventory induced by v = a(t) q / ETA.
# Integrate the regular correction separately from the singular term.
s_path = np.linspace(0, 1, 6001)
with torch.no_grad():
    f_path = model(torch.tensor(s_path[:, None])).numpy().ravel()

integral = cumulative_trapezoid(s_path * f_path, s_path, initial=0)
q_predicted = Q0 * s_path * np.exp(-(integral[-1] - integral))
q_exact = Q0 * np.sinh(K * s_path) / np.sinh(K)

times = T * (1 - s_path[::-1])
q_predicted = q_predicted[::-1]
q_exact = q_exact[::-1]

assert np.all(np.isfinite(q_predicted))
assert abs(q_predicted[0] - Q0) < 1e-8
assert abs(q_predicted[-1]) < 1e-8
assert np.all(np.diff(q_predicted) <= 1e-8)

metrics = {
    "seed": 42,
    "training_seconds": training_seconds,
    "test_normalized_residual_rmse": float(
        np.sqrt(np.mean(test_residual**2))
    ),
    "max_relative_value_coefficient_error": float(relative_error.max()),
    "predicted_initial_value_eur": float(a_predicted[-1] * Q0**2),
    "exact_initial_value_eur": float(a_exact[-1] * Q0**2),
    "max_inventory_error_shares": float(
        np.max(np.abs(q_predicted - q_exact))
    ),
}

print("\nEvaluation independante :")
for name, value in metrics.items():
    print(f"{name} : {value:.8g}")

root = Path(__file__).resolve().parents[1]
for folder in ("figures", "results", "models"):
    (root / folder).mkdir(exist_ok=True)

(root / "results" / "pinn_hjb_metrics.json").write_text(
    json.dumps(metrics, indent=2) + "\n"
)
torch.save(
    {
        "state_dict": model.state_dict(),
        "T": T, "eta": ETA, "rho": RHO, "seed": 42,
    },
    root / "models" / "pinn_hjb.pt",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].semilogy(np.arange(1, len(history) + 1), history)
axes[0].set(
    title="Adam training",
    xlabel="Iteration",
    ylabel="Normalized residual MSE",
)

axes[1].plot(times, q_exact, "--", color="black", label="Analytical")
axes[1].plot(times, q_predicted, label="Reduced HJB PINN")
axes[1].set(
    title="Induced inventory",
    xlabel="Time (minutes)",
    ylabel="Inventory (shares)",
)
axes[1].legend()

axes[2].plot(times, q_predicted - q_exact)
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set(
    title="Inventory error",
    xlabel="Time (minutes)",
    ylabel="PINN minus analytical (shares)",
)

for ax in axes:
    ax.grid(alpha=0.3)

fig.tight_layout()
figure = root / "figures" / "pinn_hjb.png"
fig.savefig(figure, dpi=160)
plt.close(fig)
print(f"\nGraphique : {figure}")
