import torch.multiprocessing as mp
import time
import queue

# Adjust these imports based on your project structure
# I'm assuming the paths from the files you provided.
from src.evaluator.worker import worker_main
from src.model.parallel import WorkerConfig, Task
from src.model.chromosome import OptimizerSchedule, AugmentationIntensity

from torchvision import datasets
from src.logger.logger import logger


def prepare_dataset():
    """
    Downloads the CIFAR-10 dataset to the './model_data' directory if it doesn't exist.
    This should be called once from the main process before any workers are spawned
    to prevent race conditions and deadlocks during download.
    """
    print("Verifying CIFAR-10 dataset...")
    try:
        # Instantiate both train and test sets with download=True.
        # This will download the files if they are missing and do nothing if they exist.
        datasets.CIFAR10(root="./model_data", train=True, download=True)
        datasets.CIFAR10(root="./model_data", train=False, download=True)
        print("Dataset is present and ready.")
    except Exception as e:
        print(f"Fatal error: Failed to download or verify the dataset: {e}")
        print("Please check your network connection and directory permissions.")
        # This is a fatal error, so we should exit.
        raise SystemExit(1)


def run_debug():
    """
    Initializes and runs a single worker process to debug potential deadlocks.
    """
    print("--- Starting Worker Debug Script ---")

    # Ensure the multiprocessing context is the same as your main app
    # This is crucial for consistent behavior.
    try:
        mp.set_start_method("spawn", force=True)
        print("[DEBUG] Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("[DEBUG] Multiprocessing context already set.")

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    prepare_dataset()
    # 1. Configure the Worker
    # This simulates the configuration for a single CPU worker.
    # We are using 2 dataloader workers as recommended.
    worker_config = WorkerConfig(
        worker_id=99,  # Use a distinct ID for debugging
        device="cpu",
        task_queue=task_queue,
        result_queue=result_queue,
        session_log_filename="debug_log.log",
        num_dataloader_workers=1,
    )
    print(
        f"[DEBUG] Created WorkerConfig for worker {worker_config.worker_id} on device '{worker_config.device}'."
    )
    print(f"[DEBUG] Dataloader workers: {worker_config.num_dataloader_workers}")

    # 2. Create a single Task with the full configuration
    task = Task(
        index=0,
        pop_size=1,
        neural_network_config={
            "input_shape": [3, 32, 32],
            "output_classes": 10,
            "conv_blocks": 3,
            "fixed_parameters": {
                "activation_function": "relu",
                "base_filters": 96,
            },
            "hyperparameter_space": {
                "width_scale": {"type": "float", "range": [1.0, 1.5]},
                "mixup_alpha": {"type": "float", "range": [0.0, 1.0]},
                "dropout_rate": {"type": "float", "range": [0.2, 0.5]},
                "optimizer_schedule": {
                    "type": "enum",
                    "values": ["SGD_ONECYCLE", "SGD_COSINE", "SGD_EXPONENTIAL"],
                },
                "base_lr": {
                    "type": "float",
                    "range": [0.005, 0.1],
                    "scale": "log",
                },
                "aug_intensity": {
                    "type": "enum",
                    "values": ["MEDIUM", "STRONG"],
                },
                "weight_decay": {
                    "type": "float",
                    "range": [1e-4, 1e-3],
                    "scale": "log",
                },
                "batch_size": {"type": "enum", "values": [128, 256]},
            },
        },
        individual_hyperparams={
            "width_scale": 1.0,
            "mixup_alpha": 0.1,
            "dropout_rate": 0.2,
            "optimizer_schedule": "SGD_COSINE",
            "base_lr": 0.01,
            "aug_intensity": "MEDIUM",
            "weight_decay": 0.0001,
            "batch_size": 128,
        },
        training_epochs=1,  # Use a small number of epochs for a quick test
        early_stop_epochs=1,
        subset_percentage=1.0,  # Use a small subset of data to speed up test
        is_final=False,
        train_indices=None,
        test_indices=None,
    )
    print(
        f"[DEBUG] Created dummy Task with {task.training_epochs} epochs and {task.subset_percentage*100}% of data."
    )

    # 3. Put the task on the queue
    task_queue.put(task)
    print("[DEBUG] Task placed in the task queue.")

    # 4. Run the worker
    # We will run the worker in a separate process to closely mimic the real environment.
    print("\n--- Launching Worker Process ---")
    print(
        "If the script hangs here, the deadlock is happening inside worker_main."
    )

    worker_process = mp.Process(target=worker_main, args=(worker_config,))
    worker_process.start()

    # 5. Wait for the result
    print("\n--- Waiting for Result ---")
    print("Waiting for the worker to complete the task. Max wait: 5 minutes.")

    try:
        # Wait for a result with a timeout. If it hangs, this will eventually terminate.
        result = result_queue.get(timeout=600)
        print("\n--- Result Received! ---")
        print(f"Status: {result.status}")
        print(f"Fitness: {result.fitness}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")

        print("\n[SUCCESS] The worker process completed successfully.")

    except queue.Empty:
        print("\n[FAILURE] Timed out waiting for a result.")
        print("The worker process is likely deadlocked or stuck.")
        print(
            "Please check the console for any errors from the worker process."
        )

    finally:
        # Clean up the worker process
        if worker_process.is_alive():
            print("Terminating hanging worker process...")
            worker_process.terminate()
            worker_process.join(timeout=5)
            if worker_process.is_alive():
                worker_process.kill()

        print("\n--- Debug Script Finished ---")


if __name__ == "__main__":
    run_debug()
