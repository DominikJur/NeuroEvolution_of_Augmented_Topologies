from src.wrapper import test_best_genome
from src.visual import visualize_best_genome


if __name__ == "__main__":
    filename = "best_genome.pkl"
    print(f"Testing best genome from {filename}...")
    print("Visualizing best genome...") 
    try:
        visualize_best_genome(filename=filename)
        test_best_genome(headless=False, filename=filename)

    except KeyboardInterrupt:
        print("Game interrupted by user.")
        sys.exit(0)
    