from pathlib import Path
import csv
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
script = root / "src" / "pinn_coupled.py"
source = script.read_text()

# Allow different seeds and preserve each run's outputs.
if "COUPLED_SEED" not in source:
    replacements = {
        "import json": "import json\nimport os",
        "torch.manual_seed(42)": (
            'SEED = int(os.environ.get("COUPLED_SEED", "42"))\n'
            'TAG = f"_seed_{SEED}" if "COUPLED_SEED" in os.environ else ""\n'
            "torch.manual_seed(SEED)"
        ),
        '"seed": 42': '"seed": SEED',
        '"pinn_coupled.json"': 'f"pinn_coupled{TAG}.json"',
        '"pinn_coupled_paths.csv"': 'f"pinn_coupled_paths{TAG}.csv"',
        '"pinn_coupled.png"': 'f"pinn_coupled{TAG}.png"',
    }
    for old, new in replacements.items():
        assert old in source, f"Texte introuvable : {old}"
        source = source.replace(old, new)
    script.write_text(source)

# Preserve the existing seed-42 result.
existing = root / "results" / "pinn_coupled.json"
baseline = json.loads(existing.read_text())
assert baseline["seed"] == 42
metrics = [baseline]

for seed in (0, 1):
    print(f"\nEntrainement couple : graine {seed}", flush=True)
    environment = os.environ.copy()
    environment["COUPLED_SEED"] = str(seed)
    subprocess.run(
        [sys.executable, str(script)],
        env=environment,
        check=True,
    )
    path = root / "results" / f"pinn_coupled_seed_{seed}.json"
    metrics.append(json.loads(path.read_text()))

rows = []
for result in sorted(metrics, key=lambda item: item["seed"]):
    rows.append({
        "seed": result["seed"],
        "first_exit_min": result["learned_first_exit_min"],
        "max_mean_inventory_error_shares":
            result["max_mean_inventory_error_shares"],
        "max_group_inventory_error_shares":
            max(result["max_group_inventory_errors_shares"]),
        "minimum_sampled_selling_rate":
            result["minimum_sampled_selling_rate"],
        "normalized_active_residual_rmse":
            result["normalized_active_residual_rmse"],
    })

output = root / "results" / "coupled_seed_validation.csv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\nBILAN DES TROIS GRAINES")
for row in rows:
    print(
        f"seed={row['seed']} | "
        f"sortie={row['first_exit_min']:.5f} min | "
        f"erreur moyenne max={row['max_mean_inventory_error_shares']:.6f} actions | "
        f"erreur groupe max={row['max_group_inventory_error_shares']:.6f} actions | "
        f"vente minimale={row['minimum_sampled_selling_rate']:.3e} | "
        f"residu={row['normalized_active_residual_rmse']:.3e}"
    )

print(f"\nResultats : {output}")
