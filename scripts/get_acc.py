import re
import sys
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(log_path: Path):
    """
    Parse one log file.
    Each '--- Starting MAIN_ALGORITHM Phase ---' begins a new sequence.
    Returns a list of (generations, fitness_values, loss_values, label_suffix)
    """
    pattern = re.compile(r"Best Fitness:\s*([0-9.]+)\s*\|\s*Loss:\s*([0-9.]+)")

    all_runs = []
    fitness_values = []
    loss_values = []
    generations = []
    generation_counter = 0
    phase_counter = 0
    main_phase_started = False

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "--- Starting MAIN_ALGORITHM Phase ---" in line:
                # Save previous phase before starting new one
                if fitness_values:
                    all_runs.append(
                        (
                            generations,
                            fitness_values,
                            loss_values,
                            f"phase_{phase_counter}",
                        )
                    )
                    fitness_values, loss_values, generations = [], [], []
                    generation_counter = 0
                phase_counter += 1
                main_phase_started = True
                continue

            if not main_phase_started:
                continue

            match = pattern.search(line)
            if match:
                fitness = float(match.group(1))
                loss = float(match.group(2))

                # 👇 Ensure non-decreasing best fitness (flat if not improved)
                if fitness_values:
                    best_so_far = fitness_values[-1]
                    if fitness < best_so_far:
                        fitness = best_so_far

                fitness_values.append(fitness)
                loss_values.append(loss)
                generations.append(len(fitness_values))
                generation_counter += 1

    # Add final phase if not empty
    if fitness_values:
        all_runs.append(
            (generations, fitness_values, loss_values, f"phase_{phase_counter}")
        )

    return all_runs


def main(log_files):
    # Use non-GUI backend if needed
    if not matplotlib.get_backend().lower().startswith("qt"):
        matplotlib.use("Agg")

    plt.figure(figsize=(10, 6))

    for log_path in log_files:
        log_path = Path(log_path)
        runs = parse_log(log_path)

        if not runs:
            print(f"⚠️ No fitness data found in {log_path}")
            continue

        for generations, fitness, loss, phase_label in runs:
            label = f"{log_path.stem} [{phase_label}] (max={max(fitness):.4f})"
            plt.plot(generations, fitness, marker="o", linewidth=2, label=label)

    plt.title("Genetic Algorithm: Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (non-decreasing)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_file = "fitness_comparison_phases.png"
    plt.savefig(output_file)
    print(f"✅ Plot saved as '{output_file}'")

    try:
        plt.show(block=True)
    except Exception:
        print("⚠️ GUI not available — plot saved as PNG instead.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_ga_fitness_multi_phase.py <log1> <log2> ...")
        sys.exit(1)

    main(sys.argv[1:])
