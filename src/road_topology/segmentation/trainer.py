"""Training utilities for segmentation models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from road_topology.core.device import empty_cache, get_autocast_context, get_device, get_grad_scaler
from road_topology.core.logging import get_logger, log_duration

logger = get_logger(__name__)


class SegmentationTrainer:
    """Trainer for semantic segmentation models.

    Features:
    - Mixed precision training (AMP)
    - Automatic checkpointing
    - Early stopping
    - Learning rate scheduling
    - Progress tracking with tqdm

    Args:
        model: Segmentation model to train.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to train on.
        scheduler: Optional learning rate scheduler.
        mixed_precision: Whether to use mixed precision training.
        gradient_clip: Maximum gradient norm for clipping (None to disable).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "auto",
        scheduler: LRScheduler | None = None,
        mixed_precision: bool = True,
        gradient_clip: float | None = 1.0,
    ) -> None:
        self.device = str(get_device(device))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip

        # Initialize mixed precision scaler (CUDA only)
        self.scaler = get_grad_scaler(self.device) if mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        logger.info(
            "Trainer initialized",
            device=device,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass with mixed precision
            with get_autocast_context(self.device, enabled=self.mixed_precision):
                outputs = self.model(pixel_values=images, labels=masks)
                loss = outputs.get("loss") or self.criterion(outputs["logits"], masks)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip,
                    )
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        logger.info(f"Training epoch {epoch} complete", avg_loss=avg_loss)

        return {"loss": avg_loss}

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Validate on validation set.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [val]")
        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                with get_autocast_context(self.device, enabled=self.mixed_precision):
                    outputs = self.model(pixel_values=images, labels=masks)
                    loss = outputs.get("loss") or self.criterion(outputs["logits"], masks)

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        logger.info(f"Validation epoch {epoch} complete", avg_loss=avg_loss)

        return {"loss": avg_loss}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Path,
        early_stopping_patience: int = 10,
        val_every: int = 1,
    ) -> dict[str, Any]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            checkpoint_dir: Directory to save checkpoints.
            early_stopping_patience: Number of epochs to wait before early stopping.
            val_every: Validate every N epochs.

        Returns:
            Training history dictionary.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        logger.info(
            "Starting training",
            num_epochs=num_epochs,
            checkpoint_dir=str(checkpoint_dir),
        )

        with log_duration(logger, "training", num_epochs=num_epochs):
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                history["train_loss"].append(train_metrics["loss"])

                # Validate
                if epoch % val_every == 0:
                    val_metrics = self.validate(val_loader, epoch)
                    history["val_loss"].append(val_metrics["loss"])

                    # Check for improvement
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.epochs_without_improvement = 0

                        # Save best checkpoint
                        self.save_checkpoint(
                            checkpoint_dir / "best_model.pt",
                            epoch=epoch,
                            is_best=True,
                            metrics=val_metrics,
                        )
                        logger.info(
                            "New best model saved",
                            epoch=epoch,
                            val_loss=val_metrics["loss"],
                        )
                    else:
                        self.epochs_without_improvement += 1

                    # Early stopping
                    if self.epochs_without_improvement >= early_stopping_patience:
                        logger.info(
                            "Early stopping triggered",
                            patience=early_stopping_patience,
                            best_val_loss=self.best_val_loss,
                        )
                        break

                # Step scheduler
                if self.scheduler:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    history["lr"].append(current_lr)

                # Save periodic checkpoint
                if epoch % 5 == 0:
                    self.save_checkpoint(
                        checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                        epoch=epoch,
                    )

        logger.info(
            "Training complete",
            final_epoch=self.current_epoch,
            best_val_loss=self.best_val_loss,
        )

        return history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        is_best: bool = False,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
            metrics: Optional metrics to save.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "is_best": is_best,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")

    def load_checkpoint(
        self,
        path: Path,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint.
            load_optimizer: Whether to restore optimizer state.

        Returns:
            Checkpoint dictionary.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load to CPU first for cross-platform compatibility (CUDA→MPS)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if "scaler_state_dict" in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(
            f"Checkpoint loaded from {path}",
            epoch=self.current_epoch,
            is_best=checkpoint.get("is_best", False),
        )

        return checkpoint

    def evaluate(
        self,
        test_loader: DataLoader,
        metrics: list[Callable[[torch.Tensor, torch.Tensor], float]] | None = None,
    ) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_loader: Test data loader.
            metrics: Optional list of metric functions.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        results = {"loss": 0.0}
        num_batches = 0

        # Initialize metric accumulators
        if metrics:
            for metric_fn in metrics:
                results[metric_fn.__name__] = 0.0

        pbar = tqdm(test_loader, desc="Evaluating")
        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                with get_autocast_context(self.device, enabled=self.mixed_precision):
                    outputs = self.model(pixel_values=images, labels=masks)
                    loss = outputs.get("loss") or self.criterion(outputs["logits"], masks)

                results["loss"] += loss.item()

                # Compute additional metrics
                if metrics:
                    preds = outputs["logits"].argmax(dim=1)
                    for metric_fn in metrics:
                        score = metric_fn(preds, masks)
                        results[metric_fn.__name__] += score

                num_batches += 1

        # Average all metrics
        for key in results:
            results[key] /= num_batches

        logger.info("Evaluation complete", **results)

        return results
