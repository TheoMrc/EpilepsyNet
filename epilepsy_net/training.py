"""
Training of EpilepsyNet model to predict moving state and seizure behavior from behavioral time series data.
Uses gradient accumulation to handle variable-length samples.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from epilepsy_net.load_data import (
    FEATURES,
    LABEL_COLS,
    TimeSeriesDataset,
    train_test_split_hash,
)
from epilepsy_net.metrics import compute_f1_metrics, compute_latency
from epilepsy_net.models import DEVICE, EpilepsyNet
from epilepsy_net.utils import DATA_DIR, plot_pred_and_labels

SAVE_DIR = "models"
PATIENCE = 30
CBM_WEIGHT = 1.0
MIN_DURATION = 5


@dataclass
class EarlyStopping:
    """Tracks best validation metrics."""

    best_val_loss: float = float("inf")
    best_val_f1: float = -float("inf")
    best_val_cbm_f1: float = -float("inf")
    best_avg_recall: float = -float("inf")
    best_hybrid: float = -float("inf")
    epochs_without_improvement: int = 0


def parse_args():
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description="Train EpilepsyNet")
    parser.add_argument(
        "--config", default="configs/training.json", help="Path to JSON config"
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--annotations-dir", default="annotations")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_run_name(config: dict) -> str:
    """Generate a descriptive run name from the selected configuration."""
    return (
        f"{config['N_LAYERS']}lyrs_{config['HIDDEN_CHANNELS']}ch_ks{config['KERNEL_SIZE']}_"
        f"batch{config['BATCH_SIZE']}_drop{config['DROPOUT_RATE']:.2f}_seed{config['SEED']}"
    ).replace(".", "_")


def run_training(config: dict, train_data, val_data, save_plots: bool = True):
    """Run training for a single hyperparameter configuration with gradient accumulation."""
    accumulation_steps = config["BATCH_SIZE"]
    learning_rate = config["LEARNING_RATE"]
    epochs = config["EPOCHS"]
    hidden_channels = config["HIDDEN_CHANNELS"]
    n_layers = config["N_LAYERS"]
    kernel_size = config["KERNEL_SIZE"]
    dropout_rate = config["DROPOUT_RATE"]

    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(train_data)
    val_dataset = TimeSeriesDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(
        f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}"
    )

    # Model and optimizer
    run_name = get_run_name(config)
    run_dir = os.path.join(SAVE_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    model = EpilepsyNet(
        features=FEATURES,
        hidden_channels=hidden_channels,
        num_layers=n_layers,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        n_output_states=len(LABEL_COLS),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1, CBM_WEIGHT, 1]).to(DEVICE)
    ).to(DEVICE)
    early_stop = EarlyStopping()

    # Print parameter count
    param_cnt = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {param_cnt:,}")

    config["train_samples"] = len(train_dataset)
    config["val_samples"] = len(val_dataset)
    config["n_parameters"] = param_cnt
    config["features"] = FEATURES
    config["n_features"] = len(FEATURES)
    config["label_cols"] = LABEL_COLS

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # WandB init
    wandb.init(
        mode=config.get("WANDB_MODE", "disabled"),
        project="epilepsynet",
        name=run_name,
        config=config,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        train_labels = []
        train_preds = []

        # Accumulation loop
        for idx, (features, labels) in enumerate(
            tqdm(
                train_loader,
                desc=f"Train Epoch {epoch}/{epochs}",
                ascii="->",
                leave=False,
            ),
            start=1,
        ):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)
            loss = (
                loss_fn(
                    logits.transpose(1, 2).squeeze(0), labels.transpose(1, 2).squeeze(0)
                )
                / accumulation_steps
            )
            loss.backward()

            train_loss += loss.item() * accumulation_steps

            # Collect metrics
            train_labels.append(labels.cpu().numpy())
            train_pred = logits.argmax(dim=1).cpu().numpy()
            train_pred_binary = np.zeros(logits.shape[1:], dtype=bool)
            for ch, _state in enumerate(LABEL_COLS):
                train_pred_binary[ch] = train_pred == ch
            train_preds.append(train_pred_binary)

            # Step and zero grads every acc_steps
            if idx % accumulation_steps == 0 or idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # Compute aggregated train metrics
        train_loss /= len(train_loader)
        concat_preds = np.concatenate(train_preds, axis=-1)
        concat_labels = np.concatenate(train_labels, axis=-1)[0].astype(bool)
        train_precision, train_recall, train_f1 = compute_f1_metrics(
            concat_preds, concat_labels
        )
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []
        proba_preds = []
        val_latency_true = []
        val_latency_pred = []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validate", leave=False):
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                logits = model(features)
                val_loss += loss_fn(
                    logits.transpose(1, 2).squeeze(0), labels.transpose(1, 2).squeeze(0)
                ).item()
                val_labels.append(labels.cpu().numpy())
                proba_preds.append(torch.softmax(logits, dim=1).cpu().numpy())

                val_pred = logits.argmax(dim=1).cpu().numpy()
                val_pred_binary = np.zeros(logits.shape[1:], dtype=bool)
                for ch, _state in enumerate(LABEL_COLS):
                    val_pred_binary[ch] = val_pred == ch
                val_preds.append(val_pred_binary)

                # Compute per-fish latency
                not_moving_label = labels[0, 0].cpu().numpy()  # not_moving channel
                not_moving_pred = val_pred_binary[0].astype(
                    float
                )  # not_moving predictions
                true_latency = compute_latency(
                    (1 - not_moving_label), min_duration=MIN_DURATION
                )
                pred_latency = compute_latency(
                    (1 - not_moving_pred), min_duration=MIN_DURATION
                )
                val_latency_true.append(true_latency)
                val_latency_pred.append(pred_latency)

        val_loss /= len(val_loader)
        concat_val_preds = np.concatenate(val_preds, axis=-1)
        concat_val_labels = np.concatenate(val_labels, axis=-1)[0].astype(bool)
        val_precision, val_recall, val_f1 = compute_f1_metrics(
            concat_val_preds, concat_val_labels
        )

        # Compute latency recall metrics
        latency_thresholds = [1, 3, 5, 10, 20]
        latency_recalls = {}
        total = len(val_latency_true)
        for thr in latency_thresholds:
            hits = 0
            for lat_true, lat_pred in zip(
                val_latency_true, val_latency_pred, strict=True
            ):
                if np.isnan(lat_true) and np.isnan(lat_pred):
                    hits += 1
                elif not np.isnan(lat_pred) and abs(lat_true - lat_pred) <= thr:
                    hits += 1
            latency_recalls[f"latency_recall@{thr}"] = hits / total

        # Compute average recall @1,3,5
        avg_recall = np.mean(
            [latency_recalls[f"latency_recall@{thr}"] for thr in [3, 5, 10]]
        )

        # Compute average over/under-estimation and rates
        val_latency_true = np.array(val_latency_true)
        val_latency_pred = np.array(val_latency_pred)
        latency_diff = np.abs(val_latency_true - val_latency_pred)
        over_estimations = val_latency_pred > val_latency_true
        under_estimations = val_latency_pred < val_latency_true

        avg_over = (
            np.nanmean(latency_diff[over_estimations])
            if np.sum(over_estimations) > 0
            else 0
        )
        avg_under = (
            np.nanmean(latency_diff[under_estimations])
            if np.sum(under_estimations) > 0
            else 0
        )
        over_rate = np.mean(over_estimations)
        under_rate = np.mean(under_estimations)
        false_positive_reactions = np.mean(
            np.isnan(val_latency_true) & ~np.isnan(val_latency_pred)
        )
        false_negative_reactions = np.mean(
            ~np.isnan(val_latency_true) & np.isnan(val_latency_pred)
        )

        # Logging
        log_dict = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "avg_recall_1_3_5": avg_recall,
            **latency_recalls,
            "average_over_estimation": avg_over,
            "average_under_estimation": avg_under,
            "over_estimation_rate": over_rate,
            "under_estimation_rate": under_rate,
            "false_positive_reactions": false_positive_reactions,
            "false_negative_reactions": false_negative_reactions,
        }

        for ch, state in enumerate(LABEL_COLS):
            precision, recall, f1 = compute_f1_metrics(
                concat_val_preds[ch], concat_val_labels[ch]
            )
            log_dict.update(
                {
                    f"val_{state}_precision": precision,
                    f"val_{state}_recall": recall,
                    f"val_{state}_f1": f1,
                }
            )

        # Compute hybrid metric: mean of val_cbm_f1 and avg_recall
        val_cbm_f1 = log_dict["val_cbm_f1"]
        hybrid = 0.5 * (val_cbm_f1 + avg_recall)
        log_dict["hybrid"] = hybrid

        wandb.log(log_dict)
        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_f1 {val_f1:.4f} | val_cbm_f1 {val_cbm_f1:.4f} | avg_recall {avg_recall:.4f} | hybrid {hybrid:.4f}"
        )

        improved = False

        if val_loss < early_stop.best_val_loss:
            early_stop.best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "best_val_loss.pt"))
            with open(os.path.join(run_dir, "best_val_loss_metrics.json"), "w") as f:
                json.dump(log_dict, f, indent=2)
            print(f"Saved best_val_loss at epoch {epoch}")
            improved = True

        if val_f1 > early_stop.best_val_f1:
            early_stop.best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(run_dir, "best_val_f1.pt"))
            with open(os.path.join(run_dir, "best_val_f1_metrics.json"), "w") as f:
                json.dump(log_dict, f, indent=2)
            print(f"Saved best_val_f1 at epoch {epoch}")
            improved = True

        if val_cbm_f1 > early_stop.best_val_cbm_f1:
            early_stop.best_val_cbm_f1 = val_cbm_f1
            torch.save(model.state_dict(), os.path.join(run_dir, "best_val_cbm_f1.pt"))
            with open(os.path.join(run_dir, "best_val_cbm_f1_metrics.json"), "w") as f:
                json.dump(log_dict, f, indent=2)
            print(f"Saved best_val_cbm_f1 at epoch {epoch}")
            improved = True

        if avg_recall > early_stop.best_avg_recall:
            early_stop.best_avg_recall = avg_recall
            torch.save(model.state_dict(), os.path.join(run_dir, "best_avg_recall.pt"))
            with open(os.path.join(run_dir, "best_avg_recall_metrics.json"), "w") as f:
                json.dump(log_dict, f, indent=2)
            print(f"Saved best_avg_recall at epoch {epoch}")
            improved = True

        if hybrid > early_stop.best_hybrid:
            early_stop.best_hybrid = hybrid
            torch.save(model.state_dict(), os.path.join(run_dir, "best_hybrid.pt"))
            with open(os.path.join(run_dir, "best_hybrid_metrics.json"), "w") as f:
                json.dump(log_dict, f, indent=2)
            print(f"Saved best_hybrid at epoch {epoch}")
            improved = True

        if improved:
            early_stop.epochs_without_improvement = 0
        else:
            early_stop.epochs_without_improvement += 1
            if early_stop.epochs_without_improvement >= PATIENCE:
                print(
                    f"No improvement for {PATIENCE} epochs, stopping training at epoch {epoch}."
                )
                break

        if save_plots and improved and epoch > 10:
            plot_path = os.path.join(run_dir, "val_plot.jpg")
            plot_pred_and_labels(
                proba_preds, val_labels, n_examples=5, file_path=plot_path
            )
            wandb.log(
                {
                    "val_plots": wandb.Image(
                        plot_path,
                        caption=f"Epoch {epoch} improvements | val_loss {val_loss:.4f} | "
                        f"val_f1 {val_f1:.4f} | val_cbm_f1 {val_cbm_f1:.4f} | avg_recall {avg_recall:.4f} | hybrid {hybrid:.4f}",
                    )
                }
            )

    # Save the last epoch model
    last_model_path = os.path.join(run_dir, f"{run_name}_last_epoch.pt")
    torch.save(model.state_dict(), last_model_path)

    artifact = wandb.Artifact("epilepsynet-models", type="model")
    for metric in ["val_loss", "val_f1", "val_cbm_f1", "avg_recall", "hybrid"]:
        model_path = os.path.join(run_dir, f"best_{metric}.pt")
        if os.path.exists(model_path):
            artifact.add_file(model_path)
    artifact.add_file(last_model_path)
    wandb.log_artifact(artifact)

    wandb.finish()


def main():
    """Entry point: parse the selected configuration and train the model."""
    global SAVE_DIR

    args = parse_args()
    SAVE_DIR = args.output_dir
    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)
    set_seed(config["SEED"])
    all_data = TimeSeriesDataset.load_dataset(args.data_dir, args.annotations_dir)
    train_data, val_data = train_test_split_hash(all_data, test_size=0.2)
    run_training(config, train_data, val_data, save_plots=True)


if __name__ == "__main__":
    main()
