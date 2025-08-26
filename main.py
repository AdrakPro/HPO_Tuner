import sys
from pathlib import Path

from src.config.settings import ex
from src.model.chromosome import (
    Chromosome,
    AugmentationIntensity,
    BatchSize,
    OptimizerSchedule,
)
from src.nn.train_and_eval import train_and_eval
from src.tui import run_tui_configurator, print_final_config_panel
from src.logger.experiment_logger import logger
from src.utils.seed import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Testing with example chromosome
chromosome = Chromosome(
    width_scale=1.5,
    fc1_units=256,
    dropout_rate=0.3,
    optimizer_schedule=OptimizerSchedule.ADAMW_COSINE,
    base_lr=0.001,
    aug_intensity=AugmentationIntensity.STRONG,
    weight_decay=1e-4,
    batch_size=BatchSize.B256,
)


@ex.main
def run_optimization(_config, _run):
    print_final_config_panel(_config)

    # ... (optimization logic) ...
    train_and_eval(chromosome)
    # ...
    logger.success("Optymalizacja zakończona.")


def main():
    try:
        seed_everything()

        config_overrides = run_tui_configurator()

        if config_overrides is not None:
            # loglevel: supress Sacred built-in logger
            ex.run(
                config_updates=config_overrides, options={"--loglevel": "ERROR"}
            )
    except KeyboardInterrupt:
        logger.error("\nUżytkownik zakończył działanie programu.")
        sys.exit(0)


if __name__ == "__main__":
    main()
