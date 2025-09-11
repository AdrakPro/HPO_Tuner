import json
import os
from collections import deque
from typing import Any, Callable, Dict

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.text import Text

from src.logger.logger import logger
from src.utils.get_nested_config import get_nested_config

# --- Constants ---
PARALLEL_CONFIG = "parallel_config"
NN_CONFIG = "neural_network_config"
HYPERPARAM_SPACE = "hyperparameter_space"
GA_CONFIG = "genetic_algorithm_config"
NESTED_VALIDATION_CONFIG = "nested_validation_config"
GENETIC_OPERATORS = "genetic_operators"
CALIBRATION_PHASE = "calibration"
MAIN_ALGORITHM = "main_algorithm"


class TUI:
    """
    Manages the layout and live updates for the Text-based User Interface.
    """

    def __init__(self):
        self.console = Console()
        self._log_deque = deque(maxlen=100)

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed} of {task.total} Evals"),
            TimeRemainingColumn(),
            expand=True,
        )

        self.layout = self._create_layout()
        self.live = Live(
            self.layout, screen=True, transient=False, refresh_per_second=10
        )

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handles graceful shutdown of the TUI."""
        self.live.stop()
        if exc_type is KeyboardInterrupt:
            return True

    def build_config_panel(self, config: Dict[str, Any]) -> None:
        """
        Builds and displays a two-panel summary of the configuration.
        """
        exec_cfg = get_nested_config(config, [PARALLEL_CONFIG, "execution"], {})
        nn_cfg = get_nested_config(config, [NN_CONFIG], {})
        fixed_params = nn_cfg.get("fixed_parameters", {})
        hyperparams = nn_cfg.get(HYPERPARAM_SPACE, {})
        cal_cfg = get_nested_config(config, [GA_CONFIG, CALIBRATION_PHASE], {})
        main_cfg = get_nested_config(config, [GA_CONFIG, MAIN_ALGORITHM], {})
        nested_validation_cfg = config[NESTED_VALIDATION_CONFIG]

        execution_mode = exec_cfg.get("evaluation_mode")
        cpu_workers = exec_cfg.get("cpu_workers", 0) or 0
        gpu_workers = exec_cfg.get("gpu_workers", 0) or 0
        dl_workers = exec_cfg.get("dataloader_workers", {})
        per_gpu = dl_workers.get("per_gpu", 0)
        per_cpu = dl_workers.get("per_cpu", 0)
        total_cpu_workers = 0

        if execution_mode == "CPU":
            total_cpu_workers = cpu_workers * per_cpu
            gpu_workers = 0
        elif execution_mode == "GPU":
            total_cpu_workers = gpu_workers * per_gpu
            cpu_workers = 0
        elif execution_mode == "HYBRID":
            total_cpu_workers = (gpu_workers * per_gpu) + (
                cpu_workers * per_cpu
            )

        max_cpu_workers = os.cpu_count()

        self._is_cpu_oversubscription(total_cpu_workers, max_cpu_workers)

        folds = nested_validation_cfg["outer_k_folds"]

        if nested_validation_cfg["enabled"] and folds > 1:
            resampling_status = f"Enabled ({folds} Folds)"
        else:
            resampling_status = "Disabled"

        config_left_details = (
            f"[bold]Execution Mode[/]: {execution_mode}\n"
            f"[bold]CPU Processes[/]: {cpu_workers}\n"
            f"[bold]GPU Processes[/]: {gpu_workers}\n"
            f"[bold]Nested Resampling[/]: {resampling_status}\n\n"
            f"[bold]Total workers count with DataLoaders[/]: {total_cpu_workers} \n(per gpu={per_gpu}, per cpu={per_cpu})\n\n"
            f"[bold cyan]-- Calibration Phase --[/]\n"
            f"  [bold]Enabled[/]: {cal_cfg.get('enabled', False)}\n"
            f"  [bold]Population[/]: {cal_cfg.get('population_size')}\n"
            f"  [bold]Generations[/]: {cal_cfg.get('generations')}\n\n"
            f"[bold cyan]-- Main Phase --[/]\n"
            f"  [bold]Population[/]: {main_cfg.get('population_size')}\n"
            f"  [bold]Generations[/]: {main_cfg.get('generations')}"
        )

        hyper_str = "\n".join(
            f"  • {name}: {details.get('range') or details.get('values')}"
            for name, details in hyperparams.items()
        )
        config_right_details = (
            f"[bold]Conv Blocks[/]: {nn_cfg.get('conv_blocks')}\n"
            f"[bold]Base Filters[/]: {fixed_params.get('base_filters')}\n"
            f"[bold]Activation[/]: {fixed_params.get('activation_function')}\n\n"
            f"[bold]Hyperparameters to Tune:[/]\n{hyper_str}"
        )

        panel_left = Panel(
            config_left_details,
            title="[cyan]GA & Execution[/]",
            border_style="cyan",
        )
        panel_right = Panel(
            config_right_details,
            title="[cyan]Neural Net & Hyperparams[/]",
            border_style="cyan",
        )

        self.layout["config_row"].split_row(
            Layout(name="config_left"), Layout(name="config_right")
        )
        self.layout["config_left"].update(panel_left)
        self.layout["config_right"].update(panel_right)

        # TODO: Responsive height based on height of text (overflow issue)
        # Adjust size dynamically
        vertical_padding = 5
        left_height = len(config_left_details.split("\n"))
        right_height = len(config_right_details.split("\n"))
        self.layout["config_row"].size = (
            max(left_height, right_height) + vertical_padding
        )
        self.layout["config_row"].visible = True

        logger.info(
            f"Configuration:\n{json.dumps(config, indent=4)}",
            file_only=True,
        )

    def get_loguru_sink(self) -> Callable[[str], None]:
        """Returns a sink function that pipes loguru records to the TUI's log panel."""

        def sink(message: str):
            self._log_deque.append(Text.from_markup(message.strip()))
            self.update()

        return sink

    def _update_header(self):
        """Updates the header with a centered title."""
        title_text = Text(
            "Genetic Hyperparameter Optimization", justify="center"
        )
        self.layout["header"].update(
            Panel(title_text, style="bold blue", border_style="blue")
        )

    def ask_resume(self) -> bool:
        """Asks the user if they want to resume from a checkpoint."""
        self.console.clear()
        self.console.print(
            Panel(
                "[bold yellow]Previous session checkpoint found.[/bold yellow]\n\nDo you want to resume?",
                title="[cyan]Resume Session[/]",
                border_style="cyan",
            )
        )
        return Confirm.ask("Resume", default=True)

    def _update_log_panel(self) -> None:
        """Updates the log panel with the latest logs, stretching to bottom and scrolling if needed."""

        term_height = self.console.size.height

        header_height = self.layout["header"].size or 3
        config_height = self.layout["config_row"].size or 0
        progress_height = self.layout["progress"].size or 3
        reserved_height = header_height + config_height + progress_height

        max_logs_height = max(1, term_height - reserved_height)

        visible_logs = list(self._log_deque)[-max_logs_height:]

        log_text = Text("\n").join(visible_logs)
        content = Align.left(log_text, vertical="top")

        panel = Panel(
            content,
            title="[bold green]Logs[/]",
            border_style="green",
            box=box.MINIMAL,
            expand=True,
        )

        self.layout["logs"].size = max_logs_height
        self.layout["logs"].update(panel)

    def update(self) -> None:
        """Updates all components of the TUI."""
        if not self.live.is_started:
            return

        try:
            self._update_header()
            self._update_log_panel()
        except Exception as e:
            logger.error(f"TUI Error during update: {e}")

    def update_fold_status(self, current_fold: int, total_folds: int):
        """Updates the fold status panel in the TUI."""
        status_text = Text(
            f"Running Fold {current_fold} of {total_folds}",
            justify="center",
            style="bold yellow",
        )
        status_panel = Panel(
            status_text,
            title="[yellow]Resampling Status[/]",
            border_style="yellow",
        )
        self.layout["fold_status_row"].update(status_panel)
        self.layout["fold_status_row"].visible = True
        self.update()

    @staticmethod
    def _is_cpu_oversubscription(
        total_cpu_workers: int, max_cpu_workers: int
    ) -> None:
        if total_cpu_workers > max_cpu_workers:
            logger.warning(
                f"Total number of CPU workers ({total_cpu_workers}) exceeded available CPU resources ({max_cpu_workers}). Program may work slower due to CPU oversubscription."
            )

    def _create_layout(self) -> Layout:
        layout = Layout(name="root")

        layout.split(
            Layout(name="header", size=3),
            Layout(name="config_row", size=0, visible=False),
            Layout(name="fold_status_row", size=3, visible=False),
            Layout(name="progress", size=3),
            Layout(name="logs", ratio=1),
        )

        layout["progress"].update(
            Panel(
                self.progress,
                title="[bold magenta]Phase Progress[/]",
                border_style="magenta",
            )
        )
        layout["logs"].update(
            Panel(
                Text(""),
                title="[bold green]Logs[/]",
                border_style="green",
                box=box.MINIMAL,
                expand=True,
            )
        )
        return layout
