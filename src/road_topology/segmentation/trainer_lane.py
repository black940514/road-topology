"""Training utilities for lane instance segmentation models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from road_topology.core.device import get_autocast_context, get_device, get_grad_scaler
from road_topology.core.logging import get_logger, log_duration
from road_topology.segmentation.losses import DiscriminativeLoss

logger = get_logger(__name__)


class LaneSegmentationTrainer:
    """Trainer for lane instance segmentation.

    Combines semantic segmentation (cross-entropy + dice) with instance
    segmentation (discriminative loss) for end-to-end lane detection.

    Features:
    - Dual-head training (semantic + instance)
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Separate loss tracking
    - Automatic checkpointing

    Args:
        model: Lane segmentation model with semantic and instance heads.
        optimizer: Optimizer instance.
        semantic_criterion: Loss for semantic segmentation.
        device: Device to train on.
        scheduler: Optional learning rate scheduler.
        mixed_precision: Whether to use mixed precision training.
        gradient_clip: Maximum gradient norm for clipping.
        semantic_weight: Weight for semantic loss component.
        instance_weight: Weight for instance loss component.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        semantic_criterion: nn.Module,
        device: str = "auto",
        scheduler: LRScheduler | None = None,
        mixed_precision: bool = True,
        gradient_clip: float | None = 1.0,
        semantic_weight: float = 1.0,
        instance_weight: float = 0.5,
        gradient_accumulation_steps: int = 1,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
    ) -> None:
        self.device = str(get_device(device))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.semantic_criterion = semantic_criterion
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize instance loss
        self.instance_criterion = DiscriminativeLoss(
            delta_var=delta_var,
            delta_dist=delta_dist,
            norm=2,
            alpha=1.0,
            beta=1.0,
            gamma=0.001,
        )

        # Initialize mixed precision scaler
        self.scaler = get_grad_scaler(self.device) if mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        logger.info(
            "LaneSegmentationTrainer initialized",
            device=self.device,
            mixed_precision=mixed_precision,
            semantic_weight=semantic_weight,
            instance_weight=instance_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
        total_semantic_loss = 0.0
        total_instance_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["image"].to(self.device)
            semantic_masks = batch["semantic_mask"].to(self.device)
            instance_masks = batch["instance_mask"].to(self.device)

            # Forward pass with mixed precision
            with get_autocast_context(self.device, enabled=self.mixed_precision):
                outputs = self.model(pixel_values=images)

                # Semantic loss
                semantic_logits = outputs["semantic_logits"]  # (B, C, H, W)
                semantic_loss = self.semantic_criterion(semantic_logits, semantic_masks)

                # Instance loss
                instance_embeddings = outputs["embeddings"]  # (B, E, H, W)
                instance_loss, instance_loss_dict = self.instance_criterion(
                    instance_embeddings, instance_masks
                )

                # Combined loss
                loss = (
                    self.semantic_weight * semantic_loss +
                    self.instance_weight * instance_loss
                )

                # Scale by accumulation steps
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    if self.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip,
                        )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update metrics (scale back up for logging)
            actual_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += actual_loss
            total_semantic_loss += semantic_loss.item()
            total_instance_loss += instance_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": actual_loss,
                "sem": semantic_loss.item(),
                "inst": instance_loss.item(),
            })

        avg_loss = total_loss / num_batches
        avg_semantic_loss = total_semantic_loss / num_batches
        avg_instance_loss = total_instance_loss / num_batches

        logger.info(
            f"Training epoch {epoch} complete",
            avg_loss=avg_loss,
            semantic_loss=avg_semantic_loss,
            instance_loss=avg_instance_loss,
        )

        return {
            "loss": avg_loss,
            "semantic_loss": avg_semantic_loss,
            "instance_loss": avg_instance_loss,
        }

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
        total_semantic_loss = 0.0
        total_instance_loss = 0.0
        num_batches = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [val]")
        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                semantic_masks = batch["semantic_mask"].to(self.device)
                instance_masks = batch["instance_mask"].to(self.device)

                with get_autocast_context(self.device, enabled=self.mixed_precision):
                    outputs = self.model(pixel_values=images)

                    # Semantic loss
                    semantic_logits = outputs["semantic_logits"]
                    semantic_loss = self.semantic_criterion(semantic_logits, semantic_masks)

                    # Instance loss
                    instance_embeddings = outputs["instance_embeddings"]
                    instance_loss, _ = self.instance_criterion(
                        instance_embeddings, instance_masks
                    )

                    # Combined loss
                    loss = (
                        self.semantic_weight * semantic_loss +
                        self.instance_weight * instance_loss
                    )

                total_loss += loss.item()
                total_semantic_loss += semantic_loss.item()
                total_instance_loss += instance_loss.item()
                num_batches += 1

                pbar.set_postfix({
                    "loss": loss.item(),
                    "sem": semantic_loss.item(),
                    "inst": instance_loss.item(),
                })

        avg_loss = total_loss / num_batches
        avg_semantic_loss = total_semantic_loss / num_batches
        avg_instance_loss = total_instance_loss / num_batches

        logger.info(
            f"Validation epoch {epoch} complete",
            avg_loss=avg_loss,
            semantic_loss=avg_semantic_loss,
            instance_loss=avg_instance_loss,
        )

        return {
            "loss": avg_loss,
            "semantic_loss": avg_semantic_loss,
            "instance_loss": avg_instance_loss,
        }

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
            "train_semantic_loss": [],
            "train_instance_loss": [],
            "val_loss": [],
            "val_semantic_loss": [],
            "val_instance_loss": [],
            "lr": [],
        }

        logger.info(
            "Starting lane segmentation training",
            num_epochs=num_epochs,
            checkpoint_dir=str(checkpoint_dir),
        )

        with log_duration(logger, "training", num_epochs=num_epochs):
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                history["train_loss"].append(train_metrics["loss"])
                history["train_semantic_loss"].append(train_metrics["semantic_loss"])
                history["train_instance_loss"].append(train_metrics["instance_loss"])

                # Validate
                if epoch % val_every == 0:
                    val_metrics = self.validate(val_loader, epoch)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_semantic_loss"].append(val_metrics["semantic_loss"])
                    history["val_instance_loss"].append(val_metrics["instance_loss"])

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
            "semantic_weight": self.semantic_weight,
            "instance_weight": self.instance_weight,
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
