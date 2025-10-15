import re
import sys


def main(log_file):
    LOW_ACC_THRESHOLD = 0.6
    OVERFIT_DIFF_THRESHOLD = 0.05

    # regex patterns
    epoch_pattern = re.compile(
        r"Epoch \d+/\d+ \| Train Acc: ([0-9.]+), Train Loss: ([0-9.]+) \| Test Loss: ([0-9.]+), Test Acc: ([0-9.]+)"
    )
    individual_pattern = re.compile(
        r"Individual \d+ -> Accuracy: ([0-9.]+), Loss: ([0-9.]+)"
    )

    overfits = []
    underfits = []
    oks = []

    with open("../logs/PART1.log", "r") as f:
        train_accs, test_accs = [], []
        final_test_acc = None
        for line in f:
            # Collect epoch data
            m = epoch_pattern.search(line)
            if m:
                train_accs.append(float(m.group(1)))
                test_accs.append(float(m.group(4)))

            # Final reported test accuracy
            m2 = individual_pattern.search(line)
            if m2:
                final_test_acc = float(m2.group(1))
                if train_accs:
                    final_train_acc = train_accs[-1]
                else:
                    final_train_acc = final_test_acc

                # Compute max overfit
                max_overfit = (
                    max([t - v for t, v in zip(train_accs, test_accs)])
                    if train_accs
                    else 0
                )
                final_overfit = final_train_acc - final_test_acc

                # Categorize
                if (
                    final_train_acc < LOW_ACC_THRESHOLD
                    and final_test_acc < LOW_ACC_THRESHOLD
                ):
                    underfits.append((final_train_acc, final_test_acc))
                elif final_overfit > OVERFIT_DIFF_THRESHOLD:
                    overfits.append(
                        (
                            final_train_acc,
                            final_test_acc,
                            final_overfit,
                            max_overfit,
                        )
                    )
                else:
                    oks.append((final_train_acc, final_test_acc))

                # reset for next Individual
                train_accs, test_accs = [], []
                final_test_acc = None

    # Summary
    print(f"Total runs: {len(overfits) + len(underfits) + len(oks)}\n")

    print(f"Overfits: {len(overfits)}")
    for t_acc, v_acc, diff, max_diff in overfits:
        print(
            f"  Train: {t_acc:.3f}, Test: {v_acc:.3f}, Final Overfit: {diff:.3f}, Max Overfit: {max_diff:.3f}"
        )

    print(f"\nUnderfits: {len(underfits)}")
    for t_acc, v_acc in underfits:
        print(f"  Train: {t_acc:.3f}, Test: {v_acc:.3f}")

    print(f"\nOK runs: {len(oks)}")
    for t_acc, v_acc in oks:
        print(f"  Train: {t_acc:.3f}, Test: {v_acc:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_ga_fitness_multi_phase.py <log1> <log2> ...")
        sys.exit(1)

    main(sys.argv[1:])
