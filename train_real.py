"""
Full training and evaluation with real / synthetic datasets.
Supports Roma Taxi, T-Drive, and controlled synthetic non-stationary traces.
"""

import torch
import torch.optim as optim
import sys
from pathlib import Path

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).parent))

from prism_model import PRISM, multi_objective_loss, compute_social_welfare
from datasets import get_dataset, DOWNLOAD_INSTRUCTIONS, ensure_datasets


def train_on_dataset(
    dataset_name: str = "synthetic",
    epochs: int = 10,
    steps_per_epoch: int = 30,
    d_model: int = 128,
    lr: float = 3e-4,
    strategic_fraction: float = 0.2,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Training PRISM on '{dataset_name}' ===")
    dataset = get_dataset(dataset_name, n_workers=80, n_tasks=30)

    model = PRISM(d_model=d_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_sw = 0.0

        for step in range(steps_per_epoch):
            slot = dataset.get_slot(epoch * steps_per_epoch + step, strategic_fraction)

            worker_feats = slot["worker_feats"].to(device)
            task_feats = slot["task_feats"].to(device)
            theta = slot["theta"].to(device)

            context = {
                "loads": slot["loads"].to(device),
                "residual_capacities": slot["residual_capacities"].to(device),
                "quality": slot["quality"].to(device),
                "public_cost": torch.tensor(0.2, device=device),
                "public_quality": torch.tensor(0.75, device=device),
                "budget": torch.tensor(100.0, device=device),
                "payment_sum": torch.tensor(0.0, device=device),
                "budget_ratio": torch.tensor(0.5, device=device),
                "reputation": torch.rand(1, device=device),
            }

            outputs = model(worker_feats, task_feats, theta, context)

            # Social welfare
            sw = compute_social_welfare(
                outputs["assignment_probs"],
                slot["task_values"].to(device),
                slot["quality"].to(device),
                outputs["payments"],
            )

            loss = multi_objective_loss(
                outputs,
                target_welfare=sw.detach(),
                target_coverage=outputs["assignment_probs"].sum(-1).mean().detach(),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sw += sw.item()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Loss: {epoch_loss/steps_per_epoch:.4f} | "
            f"SW: {epoch_sw/steps_per_epoch:.3f}"
        )

    return model, dataset


@torch.no_grad()
def evaluate_model(model, dataset, n_slots: int = 20, strategic_fraction: float = 0.2):
    model.eval()
    device = next(model.parameters()).device
    sws, parts, pays = [], [], []

    for t in range(n_slots):
        slot = dataset.get_slot(1000 + t, strategic_fraction)
        outputs = model(
            slot["worker_feats"].to(device),
            slot["task_feats"].to(device),
            slot["theta"].to(device),
            {
                "loads": slot["loads"].to(device),
                "residual_capacities": slot["residual_capacities"].to(device),
                "quality": slot["quality"].to(device),
                "public_cost": torch.tensor(0.2, device=device),
                "public_quality": torch.tensor(0.75, device=device),
                "budget": torch.tensor(100.0, device=device),
                "payment_sum": torch.tensor(0.0, device=device),
                "budget_ratio": torch.tensor(0.5, device=device),
                "reputation": torch.rand(1, device=device),
            },
        )
        sw = compute_social_welfare(
            outputs["assignment_probs"],
            slot["task_values"].to(device),
            slot["quality"].to(device),
            outputs["payments"],
        )
        sws.append(sw.item())
        part = (outputs["assignment_probs"].sum(0) > 0.05).float().mean().item()
        parts.append(part)
        pays.append(outputs["payments"].mean().item())

    return {
        "SW_mean": sum(sws) / len(sws),
        "SW_std": torch.tensor(sws).std().item(),
        "Participation": sum(parts) / len(parts),
        "Avg_Payment": sum(pays) / len(pays),
    }


def main():
    print(DOWNLOAD_INSTRUCTIONS)

    # Train on synthetic (always available)
    model, ds = train_on_dataset("synthetic", epochs=5, steps_per_epoch=20, seed=0)

    print("\n=== Evaluation ===")
    metrics = evaluate_model(model, ds, n_slots=15)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Try real datasets if files exist
    for name in ["roma", "tdrive"]:
        print(f"\n--- Checking {name} ---")
        try:
            real_ds = get_dataset(name)
            if hasattr(real_ds, "df") and real_ds.df is not None:
                print(f"Real {name} data detected. Running short evaluation...")
                m = evaluate_model(model, real_ds, n_slots=5)
                print(m)
            else:
                print(f"No real {name} files found (using synthetic fallback).")
        except Exception as e:
            print(f"{name} error: {e}")


if __name__ == "__main__":
    main()
