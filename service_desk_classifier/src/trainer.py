"""
src/trainer.py
--------------
Training loop with:
  - Weighted cross-entropy loss (class imbalance)
  - Early stopping
  - Gradient clipping
  - Per-epoch logging (loss + accuracy)
  - Best-model checkpointing
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─── Training History ─────────────────────────────────────────────────────────

class TrainingHistory:
    """Accumulates per-epoch metrics for later plotting / inspection."""

    def __init__(self):
        self.train_loss: List[float] = []
        self.val_loss:   List[float] = []
        self.train_acc:  List[float] = []
        self.val_acc:    List[float] = []
        self.lr:         List[float] = []

    def record(self, **kwargs):
        for key, val in kwargs.items():
            getattr(self, key).append(val)

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss":   self.val_loss,
            "train_acc":  self.train_acc,
            "val_acc":    self.val_acc,
            "lr":         self.lr,
        }


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss has not improved for *patience* epochs.
    Saves the best model weights when improvement occurs.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: Optional[dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Deep-copy state dict to CPU to avoid holding GPU memory
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d / %d", self.counter, self.patience
            )

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info("Best model weights restored (val_loss=%.4f)", self.best_loss)


# ─── Epoch Helpers ────────────────────────────────────────────────────────────

def _run_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    optimiser:  Optional[torch.optim.Optimizer] = None,
    grad_clip:  float = 1.0,
) -> Tuple[float, float]:
    """
    Run one epoch (train or eval).

    Returns
    -------
    (avg_loss, accuracy)
    """
    is_train = optimiser is not None
    model.train(is_train)

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(is_train):
        for sequences, labels in loader:
            sequences = sequences.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)

            logits = model(sequences)
            loss   = criterion(logits, labels)

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training lifecycle.

    Usage
    -----
    >>> trainer = Trainer(model, config, class_weights, device)
    >>> history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model:         nn.Module,
        config:        dict,
        class_weights: torch.Tensor,
        device:        torch.device,
    ):
        self.model   = model.to(device)
        self.config  = config
        self.device  = device

        tcfg = config["training"]
        pcfg = config["paths"]

        # ── Optimiser & scheduler ────────────────────────────────────────────
        self.optimiser = AdamW(
            model.parameters(),
            lr           = tcfg["learning_rate"],
            weight_decay = tcfg["weight_decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimiser,
            mode     = "min",
            factor   = 0.5,
            patience = max(1, tcfg["patience"] // 2),
        )

        # ── Loss (weighted for class imbalance) ──────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            weight = class_weights.to(device)
        )

        # ── Early stopping & checkpointing ───────────────────────────────────
        self.early_stop = EarlyStopping(patience=tcfg["patience"])
        self.model_dir  = Path(pcfg["model_save_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.model_dir / pcfg["best_model_name"]

        self.epochs     = tcfg["epochs"]
        self.grad_clip  = tcfg["gradient_clip"]
        self.history    = TrainingHistory()

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> TrainingHistory:
        """Train until convergence or max epochs, return training history."""

        logger.info("=" * 60)
        logger.info("Starting training  |  device=%s  |  epochs=%d", self.device, self.epochs)
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = _run_epoch(
                self.model, train_loader, self.criterion,
                self.device, self.optimiser, self.grad_clip,
            )
            val_loss, val_acc = _run_epoch(
                self.model, val_loader, self.criterion, self.device
            )

            current_lr = self.optimiser.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            self.history.record(
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=current_lr,
            )

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | "
                "train_loss=%.4f  train_acc=%.4f | "
                "val_loss=%.4f  val_acc=%.4f | "
                "lr=%.6f  (%.1fs)",
                epoch, self.epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr, elapsed,
            )

            # Print to console as well
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

            # Early stopping check (also saves best state internally)
            if self.early_stop(val_loss, self.model):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                print(f"\n Early stopping at epoch {epoch}.")
                break

        # Restore best weights and save checkpoint
        self.early_stop.restore_best(self.model)
        self._save_checkpoint()

        total_time = time.time() - start_time
        logger.info("Training finished in %.1f seconds.", total_time)
        print(f"\n Training complete in {total_time:.1f}s")
        print(f"  Best val_loss : {self.early_stop.best_loss:.4f}")
        print(f"  Model saved   : {self.best_model_path}")

        return self.history

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config":           self.config,
            },
            self.best_model_path,
        )
        logger.info("Checkpoint saved -> %s", self.best_model_path)
