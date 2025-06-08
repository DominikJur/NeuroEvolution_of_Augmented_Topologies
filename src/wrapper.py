import os
import pickle
from typing import List, Optional, Tuple
import time
import gc

import numpy as np
import pygame

from src.game import Game
from src.neat import FeedForwardNetwork, Genome, Config, PopulationWithDynamicSpeciation

from src.config import Config


class NeatGameWrapper:
    def __init__(self, 
                 max_steps: int = 3000,
                 fitness_weights: dict = None,
                 headless: bool = True):
        self.max_steps = max_steps
        self.headless = headless
        
        self.fitness_weights = fitness_weights or {
            'survival_time': 1.0,
            'score': 50.0,
            'jump_bonus': 2.0,
        }
        
        self.input_size = 8  # Updated: 3 basic + 1 gravity + 5 spike zones - 1 old = 8
        self.output_size = 1
        
    def encode_state(self, game: Game, steps: int = 0) -> List[float]:
        try:
            state = game.get_state_info()
            actual_player_y = state['player_pos'][1]
            actual_player_x = state['player_pos'][0]
            velocity = state['player_velocity']

            player_y = (actual_player_y / game.user_y) * 2 - 1
            
            distance_to_next_wall = (game.user_x - actual_player_x) if velocity > 0 else actual_player_x
            distance_to_next_wall /= game.user_x
            
            # NEW: Add gravity (critical for jump timing!)
            gravity_norm = np.clip(state['player_gravity'] / (10 * game.scale), -1, 1)
            
            # NEW: 5-zone spike encoding instead of 3 binary
            spikes = state['east_spike_pos'] if velocity > 0 else state['west_spike_pos']
            
            player_height = 32 * game.scale  # Approximate player height
            
            # Initialize all zones as safe (0)
            way_above = 0
            above = 0  
            at_level = 0
            below = 0
            way_below = 0
            
            if spikes:  # Only process if spikes exist
                for spike_y in spikes:
                    diff = spike_y - actual_player_y
                    
                    if diff < -2 * player_height*2:      # Way above player
                        way_above = 1
                    elif diff < -player_height*2:     # Above player
                        above = 1
                    elif abs(diff) <= player_height*2:   # At player level
                        at_level = 1
                    elif diff < 2 * player_height*2:     # Below player
                        below = 1
                    else:                              # Way below player
                        way_below = 1
            
            inputs = [
                distance_to_next_wall,
                player_y,
                gravity_norm,        # NEW: Critical for timing!
                way_above,          # NEW: More spatial granularity
                above,
                at_level,
                below,
                way_below,
            ]
            
            return inputs
        except Exception as e:
            print(f"Error in encode_state: {e}")
            return [0.0] * self.input_size
    
    def _get_coin_relative_position(self, game: Game, state: dict) -> Tuple[float, float]:
        if not game.coin_list:
            return 0.0, 0.0
        
        coin = game.coin_list[0]
        player_x, player_y = state['player_pos']
        
        rel_x = (coin.rect.x - player_x) / (game.user_x / 2)
        rel_y = (coin.rect.y - player_y) / (game.user_y / 2)
        
        return np.clip(rel_x, -1, 1), np.clip(rel_y, -1, 1)
    
    def decode_action_improved(self, network_output: List[float], context: dict) -> bool:
        # FIXED: Remove all overrides - let the network learn!
        return network_output[0] > 0.0
    
    def decode_action(self, network_output: List[float]) -> bool:
        return network_output[0] > 0.0
    
    def evaluate_genome(self, genome: Genome) -> float:
        try:
            network = FeedForwardNetwork.create_from_genome(genome)
            game = Game(headless=self.headless)
            game.running = True
            
            steps = 0
            initial_coins = game.coin_total
            initial_score = game.score
            jump_count = 0
            last_jump_frame = -10
            
            while steps < self.max_steps and not game.player.dead and game.running:
                state_vector = self.encode_state(game, steps)
                network_output = network.activate(state_vector)
                
                should_jump = self.decode_action_improved(network_output, {
                    'gravity': game.player.gravity,
                    'near_spikes': len(game.east_spikes + game.west_spikes) > 0,
                    'frames_since_jump': steps - last_jump_frame
                })
                
                game.handle_events()
                
                # Minimal anti-spam (only prevent physical impossibility)
                if should_jump and steps - last_jump_frame > 1:
                    game.player.jump()
                    jump_count += 1
                    last_jump_frame = steps
                if self.headless:
                    game.update_game()
                steps += 1

                if not self.headless:
                    game.draw_background()
                    game.draw_ui()
                    game.update_game()
                    game.screen.blit(game.player.image, game.player.rect)
                    pygame.display.update()
                    game.clock.tick(30)
            
            survival_time = steps
            score_gained = game.score - initial_score
            
            # FIXED: Remove survival ceiling, balance rewards
            survival_fitness = survival_time * (1 + survival_time / 2000)  # Exponential growth
            score_fitness = score_gained * 20  # Reduced from 50 to balance
            jump_fitness = min(survival_time / 10, jump_count * 1.0)  # Scale with survival
            
            # Stronger death penalty
            death_penalty = -max(100, survival_time * 0.3) if game.player.dead else 0
            
            # Simple efficiency bonus
            efficiency_bonus = 0
            if survival_time > 200 and jump_count > 0:
                efficiency_bonus = (survival_time / jump_count) * 0.2
            
            total_fitness = survival_fitness + score_fitness + jump_fitness + death_penalty + efficiency_bonus
            
            # Cleanup for long runs
            if self.headless:
                pygame.quit()
                pygame.init()
                gc.collect()
            
            return max(1.0, total_fitness)
        except Exception as e:
            print(f"Error evaluating genome {genome.id}: {e}")
            return 1.0  # Return minimum fitness on error
    
    def run_visual_test(self, genome: Genome):
        try:
            network = FeedForwardNetwork.create_from_genome(genome)
            game = Game(headless=False)
            
            print(f"Testing genome {genome.id} with {len(genome.nodes)} nodes and {len(genome.connections)} connections")
            
            game.running = True
            steps = 0
            jump_count = 0
            last_jump_frame = -10
            
            while steps < self.max_steps and not game.player.dead:
                game.handle_events()
                
                state_vector = self.encode_state(game, steps)
                network_output = network.activate(state_vector)
                
                should_jump = self.decode_action_improved(network_output, {
                    'gravity': game.player.gravity,
                    'near_spikes': len(game.east_spikes + game.west_spikes) > 0,
                    'frames_since_jump': steps - last_jump_frame
                })
                
                if should_jump and steps - last_jump_frame > 1:  # Minimal anti-spam
                    game.player.jump()
                    jump_count += 1
                    last_jump_frame = steps
                
                game.draw_background()
                game.draw_ui()
                
                if game.running:
                    game.update_game()
                
                game.screen.blit(game.player.image, game.player.rect)
                
                debug_text = f"Jumps: {jump_count}, Steps: {steps}, Score: {game.score}, Output: {network_output[0]:.3f}"
                game.draw_text(debug_text, 20 * game.scale, game.user_x // 2, 10)
                
                game.update_high_score()
                pygame.display.update()
                game.clock.tick(30)
                
                steps += 1
            
            print(f"Game ended after {steps} steps. Score: {game.score}, Jumps: {jump_count}, Dead: {game.player.dead}")
        except Exception as e:
            print(f"Error in visual test: {e}")
    
    def get_input_size(self) -> int:
        return self.input_size
    
    def get_output_size(self) -> int:
        return self.output_size
    
def save_genome(genome, filename="best_genome.pkl"):
    try:
        os.makedirs("models", exist_ok=True)
        filepath = os.path.join("models", filename)
        with open(filepath, "wb") as f:
            pickle.dump(genome, f)
        print(f"Genome saved to {filepath}")
    except Exception as e:
        print(f"Error saving genome: {e}")

def save_training_log(generation, best_fitness, species_count, filename="training_log.txt"):
    try:
        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", filename)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(filepath, "a") as f:
            f.write(f"{timestamp}, Gen {generation}, Fitness {best_fitness:.2f}, Species {species_count}\n")
    except Exception as e:
        print(f"Error saving log: {e}")

def train_neat_on_game(headless: bool = True, num_generations: int = 200) -> Genome:
    print(f"Starting NEAT training for {num_generations} generations...")
    print(f"Headless mode: {headless}")
    
    wrapper = NeatGameWrapper(
        max_steps=3000,
        fitness_weights={
            'survival_time': 1.0,
            'score': 50.0,
            'jump_bonus': 2.0
        },
        headless=headless
    )
    
    config = Config()
        
    population = PopulationWithDynamicSpeciation(
        config, 
        wrapper.get_input_size(), 
        wrapper.get_output_size()
    )
    
    def fitness_function(genome):
        try:
            # FIXED: This was completely broken before!
            fitness = wrapper.evaluate_genome(genome)
            
            # Save every 10 generations
            if population.generation % 10 == 0:
                best_individual = max(population.individuals, key=lambda x: x.fitness)
                if genome == best_individual.genome:
                    save_genome(genome, f"best_genome_gen_{population.generation}.pkl")
                    save_genome(genome, "best_genome.pkl")  # Also save as latest
                    save_training_log(population.generation, fitness, len(population.species))
            
            return fitness
        except Exception as e:
            print(f"Error in fitness function: {e}")
            return 1.0

    try:
        winner = population.run(fitness_function, num_generations=num_generations)

        print(f"Training complete! Best fitness: {winner.fitness}")
        
        # Final save
        save_genome(winner.genome, "final_best_genome.pkl")
        
        # Visual test if not too many generations
        if num_generations <= 100:
            wrapper.run_visual_test(winner.genome)

        return winner.genome
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        if population.individuals:
            best = max(population.individuals, key=lambda x: x.fitness)
            save_genome(best.genome, "interrupted_best_genome.pkl")
            print(f"Saved best genome from generation {population.generation}")
        return None
    except Exception as e:
        print(f"Training error: {e}")
        return None

def load_best_genome(filename="best_genome.pkl") -> Optional[Genome]:
    try:
        filepath = os.path.join("models", filename)
        if not os.path.exists(filepath):
            print(f"No saved genome found at {filepath}")
            return None
        
        with open(filepath, "rb") as f:
            genome = pickle.load(f)
        
        print(f"Loaded genome with ID {genome.id} and fitness {genome.fitness}")
        return genome
    except Exception as e:
        print(f"Error loading genome: {e}")
        return None

def test_best_genome(headless: bool = False):
    genome = load_best_genome()
    if genome:
        wrapper = NeatGameWrapper(headless=headless)
        wrapper.run_visual_test(genome)
        return genome

if __name__ == "__main__":
    # Ready for long training runs!
    train_neat_on_game(headless=True, num_generations=300)