import sys

from src.wrapper import train_neat_on_game

if __name__ == "__main__":
    try:
        train_neat_on_game(num_generations=250, headless=1)
    except KeyboardInterrupt:
        print("Game interrupted by user.")
        sys.exit(0)