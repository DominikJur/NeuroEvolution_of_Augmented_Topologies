from src.wrapper import test_best_genome
from src.visual import visualize_best_genome


if __name__ == "__main__":
    try:
        visualize_best_genome()
        test_best_genome(headless=False)

    except KeyboardInterrupt:
        print("Game interrupted by user.")
        sys.exit(0)
    