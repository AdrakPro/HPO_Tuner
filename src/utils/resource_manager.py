import os
from typing import Any, Dict

from src.logger.logger import logger


def adjust_worker_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically adjusts the number of workers to prevent CPU oversubscription.

    It calculates the total CPU core demand from cpu_workers and all their
    respective dataloader workers. If the demand exceeds available cores,
    it proportionally scales down the worker counts.

    Args:
        config: The initial user configuration.

    Returns:
        The adjusted (or original) configuration.
    """
    exec_config = config["parallel_config"]["execution"]

    if not exec_config["enable_parallel"]:
        return config

    available_cpus = os.cpu_count()

    evaluation_mode = exec_config["evaluation_mode"]
    cpu_procs = exec_config["cpu_workers"]
    gpu_procs = exec_config["gpu_workers"]

    if evaluation_mode == "CPU":
        gpu_procs = 0
    elif evaluation_mode == "GPU":
        cpu_procs = 0

    dl_config = exec_config.get("dataloader_workers", {})
    dl_per_cpu = dl_config.get("per_cpu", 1)
    dl_per_gpu = dl_config.get("per_gpu", 1)

    # Each CPU process uses 1 core for itself + 'dl_per_cpu' for its dataloaders.
    # Each GPU process uses 'dl_per_gpu' cores for its dataloaders.
    # Assume the main GPU process itself does not consume a full CPU core.
    cpu_demand_per_cpu_proc = 1 + dl_per_cpu
    cpu_demand_per_gpu_proc = dl_per_gpu

    total_demand = (cpu_procs * cpu_demand_per_cpu_proc) + (
        gpu_procs * cpu_demand_per_gpu_proc
    )

    if total_demand <= available_cpus:
        return config

    logger.warning(
        f"CPU oversubscription detected! Requested cores ({total_demand}) exceed available ({available_cpus})."
    )
    logger.warning("Scaling down worker configuration to fit resources.")

    scaling_factor = available_cpus / total_demand

    # Scale down the number of main processes
    adj_cpu_procs = max(1, int(cpu_procs * scaling_factor))
    adj_gpu_procs = int(
        gpu_procs * scaling_factor
    )  # Can be 0 if not enough resources

    # Scale down dataloader workers, ensuring at least 1 if possible.
    adj_dl_per_cpu = max(1, int(dl_per_cpu * scaling_factor))
    adj_dl_per_gpu = max(1, int(dl_per_gpu * scaling_factor))

    config["parallel_config"]["execution"]["cpu_workers"] = adj_cpu_procs
    config["parallel_config"]["execution"]["gpu_workers"] = adj_gpu_procs
    config["parallel_config"]["execution"]["dataloader_workers"][
        "per_cpu"
    ] = adj_dl_per_cpu
    config["parallel_config"]["execution"]["dataloader_workers"][
        "per_gpu"
    ] = adj_dl_per_gpu

    new_demand_cpu = adj_cpu_procs * (1 + adj_dl_per_cpu)
    new_demand_gpu = adj_gpu_procs * adj_dl_per_gpu
    new_total_demand = new_demand_cpu + new_demand_gpu

    logger.info("--- Adjusted Worker Configuration ---")
    logger.info(f"CPU Processes: {adj_cpu_procs} (was {cpu_procs})")
    logger.info(f"GPU Processes: {adj_gpu_procs} (was {gpu_procs})")
    logger.info(
        f"Dataloader workers/CPU process: {adj_dl_per_cpu} (was {dl_per_cpu})"
    )
    logger.info(
        f"Dataloader workers/GPU process: {adj_dl_per_gpu} (was {dl_per_gpu})"
    )
    logger.info(
        f"New estimated CPU core usage: {new_total_demand}/{available_cpus}"
    )

    return config
