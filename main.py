import sys
from pathlib import Path
from rich.console import Console
from src.config.settings import ex
from src.tui import run_tui_configurator, print_final_config_panel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

console = Console()


@ex.main
def main(_config, _run):
    console.print(
        "[bold green]Starting Genetic Hyperparameter Optimization...[/bold green]"
    )

    print_final_config_panel(_config)

    # ... (optimization logic) ...
    console.print("1. Initializing population...")
    # ...
    console.print("[bold green]Optimization finished.[/bold green]")


if __name__ == "__main__":
    try:
        config_overrides = run_tui_configurator()

        if config_overrides is not None:
            ex.run(config_updates=config_overrides)
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
        sys.exit(0)
