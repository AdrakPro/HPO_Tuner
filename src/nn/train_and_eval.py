from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.logger.logger import logger
from src.model.chromosome import Chromosome, AugmentationIntensity
from src.nn.data_loader import update_train_augmentation
from src.nn.model_saver import ModelSaver
from src.nn.neural_network import get_network, get_optimizer_and_scheduler
from src.utils.exceptions import CudaOutOfMemoryError, NumericalInstabilityError


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    scheduler = None,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: Neural network model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run computations on.
        scaler: Scaler supporting Mixed Precision

    Returns:
        Tuple of average loss and accuracy for the epoch.

    Raises:
        NumericalInstabilityError: If loss becomes NaN or infinity.
        CudaOutOfMemoryError: If a CUDA OOM error is detected.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    nan_batches = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # NaN loss check
        if not torch.isfinite(loss):
            nan_batches += 1
            if nan_batches > 5:
                raise NumericalInstabilityError("NaN loss detected multiple times.")
            continue

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # important for gradient clipping
        else:
            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # NaN gradients check
        has_nan_grad = any(param.grad is not None and torch.isnan(param.grad).any() for param in model.parameters())
        if has_nan_grad:
            nan_batches += 1
            if nan_batches > 5:
                raise NumericalInstabilityError("NaN gradients detected multiple times.")
            optimizer.zero_grad()
            continue

        # Step optimizer
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Scheduler step per batch if OneCycleLR
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total if total > 0 else float("inf")
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model.

    Args:
        model: Neural network model.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: Device to run computations on.

    Returns:
        Tuple of average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_and_eval(
    chromosome: Chromosome,
    neural_config: Dict[str, Any],
    epochs: int,
    early_stop_epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    is_final: bool = False,
    epoch_callback=None,
) -> Tuple[float, float]:
    """
    Train and evaluate CNN.

    Args:
        chromosome: Chromosome describing the model/hyperparameters.
        neural_config: Additional model config.
        epochs: Max number of training epochs.
        early_stop_epochs: Number of epochs to wait for improvement.
        device: torch device (CPU/GPU).
        train_loader: Train PyTorch's data loader
        test_loader: Test PyTorch's data loader
        is_final: Save the model if True.
        epoch_callback: Optional callback per epoch.

    Returns:
        Tuple of (final_test_accuracy, final_test_loss).
    """

    try:
        model: nn.Module = get_network(chromosome, neural_config).to(device)
        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer_and_scheduler(
            model, chromosome, train_loader, epochs
        )
        scaler = torch.amp.GradScaler() if device.type == "cuda" else None

        final_test_acc = 0.0
        final_test_loss = 0.0
        best_acc_so_far = 0.0
        epochs_without_improvement = 0

        switch_epoch = max(1, int(0.1 * epochs))

        for epoch in range(epochs):
            # STRONG augmentation on start causes neural network to not learn at all
            if (
                epoch == switch_epoch
                and chromosome.aug_intensity == AugmentationIntensity.STRONG
            ):
                update_train_augmentation(train_loader, AugmentationIntensity.STRONG)

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, scheduler
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            if scheduler:
                scheduler.step()

            final_test_acc = test_acc
            final_test_loss = test_loss

            if epoch_callback is not None:
                epoch_callback(epoch + 1, train_acc, train_loss, test_acc, test_loss)
            else:
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs} | "
                    f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                )

            if test_acc > best_acc_so_far:
                best_acc_so_far = test_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stop_epochs:
                if epoch_callback is not None:
                    epoch_callback(
                        epoch + 1,
                        train_acc,
                        train_loss,
                        test_acc,
                        test_loss,
                        early_stop=True,
                    )
                else:
                    logger.info(
                        f"  Stopping early after epoch {epoch + 1} "
                        f"due to no improvement for {early_stop_epochs} epochs."
                    )
                break

        if is_final:
            saver = ModelSaver("model")
            callback_msg = saver.save(model)
            if callback_msg:
                if epoch_callback is not None:
                    epoch_callback(None, None, None, None, None, final_msg=callback_msg)
                else:
                    logger.info(callback_msg)

        return final_test_acc, final_test_loss

    except (CudaOutOfMemoryError, NumericalInstabilityError):
        raise
    except RuntimeError as e:
        if "DataLoader worker" in str(e) and (
            "killed" in str(e) or "exited unexpectedly" in str(e)
        ):
            logger.error(
                "DataLoader worker failed. This is often due to insufficient shared memory."
            )
            logger.error(f"An unexpected runtime error occurred in train_and_eval: {e}")
            raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in train_and_eval: {e}")
        raise
