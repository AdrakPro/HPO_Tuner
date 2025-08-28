import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.logger.experiment_logger import logger
from src.model.chromosome import Chromosome
from src.nn.data_loader import get_dataset_loaders
from src.nn.neural_network import CNN, get_network, get_optimizer_and_scheduler


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: Neural network model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run computations on.

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
        outputs = model(inputs)
        loss = criterion(outputs, targets)
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

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_and_eval(
    chromosome: Chromosome, config: any, epochs: int
) -> tuple[float, float]:
    """
    Train and evaluate CNN on CIFAR-10.
    """
    is_gpu: bool = "GPU" in config["hardware_config"]["evaluation_mode"]
    padding: int = config["neural_network_config"]["fixed_parameters"][
        "padding"
    ]

    # What about when CPU+GPU
    device: torch.device = torch.device("cuda" if is_gpu else "cpu")

    try:
        train_loader, test_loader = get_dataset_loaders(
            chromosome.batch_size, chromosome.aug_intensity, is_gpu, padding
        )
    except Exception as e:
        logger.warning(
            f"Could not load real data ({e}). Using dummy data loaders."
        )
        bs = chromosome.batch_size
        train_loader = [
            (torch.randn(bs, 3, 32, 32), torch.randint(0, 10, (bs,)))
            for _ in range(5)
        ]
        test_loader = [
            (torch.randn(bs, 3, 32, 32), torch.randint(0, 10, (bs,)))
            for _ in range(2)
        ]

    model: CNN = get_network(chromosome, config["neural_network_config"]).to(
        device
    )
    criterion: nn.Module = nn.CrossEntropyLoss()

    optimizer, scheduler = get_optimizer_and_scheduler(
        model, chromosome, train_loader, epochs
    )

    final_test_acc = 0.0
    final_test_loss = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if scheduler:
            scheduler.step()

        final_test_acc = test_acc
        final_test_loss = test_loss
        print(
            f"  Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

    return final_test_acc, final_test_loss

    # saver = ModelSaver("cnn")
    # saver.save(model)
