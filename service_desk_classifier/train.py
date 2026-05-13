"""
train.py
--------
Entry point: run the full training pipeline.

Usage
-----
# Default config
python train.py

# Custom config
python train.py --config configs/config.yaml

# Override architecture
python train.py --arch textcnn --epochs 20 --lr 5e-4
"""

import argparse
import json
import sys
from pathlib import Path

# ── allow running from project root without installing the package ──
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import DataPipeline
from src.model import build_model
from src.trainer import Trainer
from src.evaluate import evaluate, plot_training_curves
from src.utils import load_config, setup_logging, set_seed, get_device, save_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Service Desk Ticket Classifier")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p.add_argument("--arch",   default=None, help="Override architecture (bilstm/textcnn)")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--lr",     type=float, default=None, help="Override learning rate")
    p.add_argument("--batch",  type=int, default=None, help="Override batch size")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    # ── CLI overrides ──────────────────────────────────────────────────────────
    if args.arch:
        config["model"]["architecture"] = args.arch
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch:
        config["training"]["batch_size"] = args.batch

    # ── Logging & reproducibility ──────────────────────────────────────────────
    lcfg = config.get("logging", {})
    setup_logging(
        level    = lcfg.get("level", "INFO"),
        log_file = lcfg.get("log_filename") if lcfg.get("log_to_file") else None,
    )
    set_seed(config["data"]["random_seed"])
    device = get_device()

    # ── Data pipeline ──────────────────────────────────────────────────────────
    print("\n[1/4] Preparing data …")
    pipeline = DataPipeline(config)
    train_loader, val_loader, test_loader = pipeline.prepare()

    vocab         = pipeline.vocab
    label_encoder = pipeline.label_encoder
    class_weights = pipeline.class_weights

    print(f"      Vocabulary size : {len(vocab):,}")
    print(f"      Classes ({len(label_encoder)})    : {label_encoder.classes}")

    # Save vocab and label encoder for inference
    Path(config["paths"]["model_save_dir"]).mkdir(parents=True, exist_ok=True)
    vocab.save(f"{config['paths']['model_save_dir']}/vocab.pkl")
    label_encoder.save(f"{config['paths']['model_save_dir']}/label_encoder.pkl")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Building model …")
    arch  = config["model"]["architecture"]
    model = build_model(arch, len(vocab), len(label_encoder), config)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[3/4] Training …\n")
    trainer = Trainer(model, config, class_weights, device)
    history = trainer.fit(train_loader, val_loader)

    # Save training curves
    plot_training_curves(
        history.to_dict(),
        save_path=f"{config['paths']['results_dir']}/training_curves.png",
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set …")
    metrics = evaluate(
        model        = model,
        loader       = test_loader,
        device       = device,
        class_names  = label_encoder.classes,
        results_dir  = config["paths"]["results_dir"],
        split_name   = "test",
    )

    # Save numeric metrics to JSON (excluding the report string)
    save_results(
        {k: v for k, v in metrics.items() if k != "classification_report"},
        path=f"{config['paths']['results_dir']}/metrics.json",
    )

    print("\nTraining completed. All artefacts saved to:", config["paths"]["results_dir"])
    print("  Model checkpoint   :", config["paths"]["model_save_dir"])


if __name__ == "__main__":
    main()
