import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.logger.experiment_logger import logger
from src.model.chromosome import Chromosome
from src.nn.data_loader import get_dataset_loaders
from src.nn.model_saver import ModelSaver
from src.nn.neural_network import CNN, get_network, get_optimizer_and_scheduler


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
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
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Mixed Precision Support
        if scaler is not None:
            with autocast(device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Numerical instability detected: loss is {loss.item()}. Assigned fitness '0.0'."
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Numerical instability detected: loss is {loss.item()}. Assigned fitness '0.0'."
                )

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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

            if not torch.isfinite(loss):
                logger.warning(
                    f"Infinite or NaN loss detected during evaluation: {loss.item()}. Assigned fitness 0.0"
                )
                running_loss += float("inf")
            else:
                running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_and_eval(
    chromosome: Chromosome,
    config: any,
    epochs: int,
    early_stop_epochs: int,
    subset_percentage: float = 1.0,
    is_final: bool = False,
) -> tuple[float, float]:
    """
    Train and evaluate CNN on CIFAR-10.
    """
    is_gpu: bool = "GPU" in config["hardware_config"]["evaluation_mode"]

    # What about when CPU+GPU
    device: torch.device = torch.device("cuda" if is_gpu else "cpu")

    try:
        with get_dataset_loaders(
            chromosome.batch_size,
            chromosome.aug_intensity,
            is_gpu,
            subset_percentage,
        ) as (train_loader, test_loader):
            model: CNN = get_network(
                chromosome, config["neural_network_config"]
            ).to(device)
            criterion: nn.Module = nn.CrossEntropyLoss()

            optimizer, scheduler = get_optimizer_and_scheduler(
                model, chromosome, train_loader, epochs
            )

            # Mixed Precision Support
            scaler = GradScaler() if is_gpu else None

            final_test_acc = 0.0
            final_test_loss = 0.0
            best_acc_so_far = 0.0
            epochs_without_improvement = 0

            for epoch in range(epochs):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, scaler
                )
                test_loss, test_acc = evaluate(
                    model, test_loader, criterion, device
                )

                if scheduler:
                    scheduler.step()

                final_test_acc = test_acc
                final_test_loss = test_loss
                # TODO: change if parallelism is on
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                )

                # Early stopping logic
                # TODO: impl median stopping (sync halving or async with shared state)
                if test_acc > best_acc_so_far:
                    best_acc_so_far = test_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stop_epochs:
                    logger.warning(
                        f"  Stopping individual early after epoch {epoch + 1} due to no improvement for {early_stop_epochs} epochs."
                    )
                    break

            if is_final:
                saver = ModelSaver("model")
                saver.save(model)

            return final_test_acc, final_test_loss
    except Exception as e:
        logger.error(f"Unexpected error: ({e}).")
        return 0.0, float("inf")
