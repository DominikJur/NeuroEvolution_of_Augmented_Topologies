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
                 headless: bool = True,
                 show_zones: bool = True):  # NEW: Zone visualization flag
        self.max_steps = max_steps
        self.headless = headless
        self.show_zones = show_zones  # NEW
        self.zone_width = 32  # Width of each zone in pixels
        
        # Set dummy video driver for headless mode BEFORE pygame init
        if self.headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        # Initialize pygame ONCE at wrapper creation
        if not pygame.get_init():
            pygame.init()
        
        self.input_size = 8
        self.output_size = 1
    
    def draw_spike_zones(self, game: Game):
        """Zone visualization overlay - FIXED to match encoding"""
        if self.headless or not self.show_zones:
            return
        
        player_y = game.player.rect.y
        zone_width = self.zone_width * game.scale

        # FIXED: Zone boundaries now match encoding exactly
        zones = [
            # way_above: diff < -3 * zone_width (everything above this line)
            (0, 0, game.user_x, max(0, player_y - 3 * zone_width), (255, 100, 100, 60)),
            
            # above: -3 * zone_width <= diff < -1 * zone_width  
            (0, max(0, player_y - 3 * zone_width), game.user_x, max(0, player_y - zone_width), (255, 150, 0, 60)),
            
            # at_level: abs(diff) <= zone_width (player_y - zone_width to player_y + zone_width)
            (0, max(0, player_y - zone_width), game.user_x, min(game.user_y, player_y + zone_width), (255, 0, 0, 100)),
            
            # below: zone_width < diff < 3 * zone_width (CORRECTED: back to 3 * zone_width) 
            (0, min(game.user_y, player_y + zone_width), game.user_x, min(game.user_y, player_y + 3 * zone_width), (100, 255, 100, 60)),
            
            # way_below: diff >= 3 * zone_width (CORRECTED: back to 3 * zone_width)
            (0, min(game.user_y, player_y + 3 * zone_width), game.user_x, game.user_y, (0, 255, 0, 60))
        ]
        
        # Draw zones
        for x, y1, width, y2, color in zones:
            if y2 > y1:
                zone_surface = pygame.Surface((width, y2 - y1), pygame.SRCALPHA)
                zone_surface.fill(color)
                game.screen.blit(zone_surface, (x, y1))

    def encode_state(self, game: Game, steps: int = 0) -> List[float]:
        """FIXED: Encoding now matches visualization exactly"""
        try:
            state = game.get_state_info()
            actual_player_y = state['player_pos'][1]
            actual_player_x = state['player_pos'][0]
            velocity = state['player_velocity']

            player_y = (actual_player_y / game.user_y) * 2 - 1
            
            distance_to_next_wall = (game.user_x - actual_player_x) if velocity > 0 else actual_player_x
            distance_to_next_wall /= game.user_x
            
            # Add gravity (critical for jump timing!)
            gravity_norm = np.clip(state['player_gravity'] / (10 * game.scale), -1, 1)
            
            # FIXED: 5-zone spike encoding - now matches visualization
            spikes = state['east_spike_pos'] if velocity > 0 else state['west_spike_pos']
            
            # Initialize all zones as safe (0)
            way_above = 0
            above = 0  
            at_level = 0
            below = 0
            way_below = 0
            
            zone_width = self.zone_width * game.scale
            
            if spikes:  # Only process if spikes exist
                for spike_y in spikes:
                    diff = spike_y - actual_player_y

                    # FIXED: Zone logic now matches visualization exactly
                    if diff < -3 * zone_width:              # Way above player
                        way_above = 1
                    elif diff < -1 * zone_width:            # Above player (-3 to -1 zone_width)
                        above = 1
                    elif abs(diff) <= zone_width:           # At player level (±zone_width)
                        at_level = 1
                    elif diff < 3 * zone_width:             # Below player (CORRECTED: back to 3 * zone_width)
                        below = 1
                    else:                                   # Way below player (>=3 * zone_width)
                        way_below = 1
            
            inputs = [
                distance_to_next_wall,
                player_y,
                gravity_norm,        # Critical for timing!
                way_above,          # Fixed 5-zone system
                above,
                at_level,
                below,
                way_below,
            ]
            
            return inputs
        except Exception as e:
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
        # Let the network learn everything - no overrides!
        return network_output[0] > 0.0
    
    def decode_action(self, network_output: List[float]) -> bool:
        return network_output[0] > 0.0
    
    def evaluate_genome(self, genome: Genome) -> float:
        try:
            network = FeedForwardNetwork.create_from_genome(genome)
            game = Game(headless=self.headless)
            game.running = True
            
            steps = 0
            distance_bonus = 0
            score = game.score
            
            while steps < self.max_steps and not game.player.dead and game.running:
                state_vector = self.encode_state(game, steps)
                network_output = network.activate(state_vector)
                
                should_jump = self.decode_action_improved(network_output, {})
                
                game.handle_events()
                
                if should_jump:
                    game.player.jump()
                
                pre_collision_wall_check = game.player.check_wall_collision()
                
                if self.headless:
                    game.update_game()
                else:
                    game.draw_background()
                    game.draw_ui()
                    game.update_game()
                    self.draw_spike_zones(game)  # NEW: Draw zones
                    game.screen.blit(game.player.image, game.player.rect)
                    pygame.display.update()
                    game.clock.tick(30)
                
                post_collision_wall_check = game.player.check_wall_collision()
                
                # REWARD: Successfully hit wall (means avoided spikes!)
                if pre_collision_wall_check == 0 and post_collision_wall_check != 0:
                    distance_bonus += self._calculate_wall_positioning_bonus(game)
                    

                steps += 1

            death_penalty = -500 if game.player.dead else 0

            total_fitness = max(1.0, steps + distance_bonus + death_penalty)

            for _ in range(game.score - score):
                total_fitness *= 1.1

            # Cleanup
            game.cleanup()
            del game
            del network
            if self.headless:
                gc.collect()
            
            return total_fitness
        except Exception as e:
            print(f"Error in evaluate_genome: {e}")
            return 1.0
    
    def _calculate_wall_positioning_bonus(self, game: Game) -> float:
        """Calculate bonus for smart positioning when hitting walls"""
        player_y = game.player.rect.y
        player_center_y = player_y + game.player.rect.height // 2
        
        current_spikes = game.east_spikes + game.west_spikes
        
        if not current_spikes:
            return 10.0 
        
        min_distance = float('inf')
        
        for spike in current_spikes:
            spike_center_y = spike.rect.y + spike.rect.height // 2
            distance = abs(player_center_y - spike_center_y)
            
            if distance < min_distance:
                min_distance = distance
        
        return min_distance
    
    def run_visual_test(self, genome: Genome):
        try:
            network = FeedForwardNetwork.create_from_genome(genome)
            game = Game(headless=False)
            
            game.running = True
            steps = 0
            last_jump_frame = -10
            
            while steps < self.max_steps and not game.player.dead:
                game.handle_events()
                
                state_vector = self.encode_state(game, steps)
                network_output = network.activate(state_vector)
                
                should_jump = self.decode_action_improved(network_output, {})
                
                if should_jump and steps - last_jump_frame > 1:
                    game.player.jump()
                    last_jump_frame = steps
                
                game.draw_background()
                game.draw_ui()
                
                if game.running:
                    game.update_game()
                
                self.draw_spike_zones(game)  # NEW: Show zones in visual test
                game.screen.blit(game.player.image, game.player.rect)
                
                debug_text = f"Steps: {steps}, Score: {game.score}, Output: {network_output[0]:.3f}"
                game.draw_text(debug_text, 20 * game.scale, game.user_x // 2, 10)
                
                # Optional: Enable debug zone encoding (uncomment to use)
                # if steps % 60 == 0:  # Print every 2 seconds at 30 FPS
                #     self.debug_zone_encoding(game)
                
                game.update_high_score()
                pygame.display.update()
                game.clock.tick(30)
                
                steps += 1
            
        except Exception as e:
            pass
    
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
    except Exception as e:
        pass


def train_neat_on_game(headless: bool = True, num_generations: int = 200) -> Genome:
    wrapper = NeatGameWrapper(
        max_steps=2000,
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
            fitness = wrapper.evaluate_genome(genome)
            
            # Periodic pygame cleanup to prevent memory leaks
            if population.generation % 50 == 0:
                pygame.quit()
                pygame.init()
            
            # Save every 10 generations
            if population.generation % 10 == 0:
                best_individual = max(population.individuals, key=lambda x: x.fitness)
                if genome == best_individual.genome:
                    save_genome(genome, f"best_genome_gen_{population.generation}.pkl")
                    save_genome(genome, "best_genome.pkl")
            
            return fitness
        except Exception as e:
            print(f"Fitness function error: {e}")
            return 1.0

    try:
        winner = population.run(fitness_function, num_generations=num_generations)
        
        # Final save
        save_genome(winner.genome, "final_best_genome.pkl")
        
        # Visual test if not too many generations
        if num_generations <= 100:
            wrapper.run_visual_test(winner.genome)

        return winner.genome
    except KeyboardInterrupt:
        if population.individuals:
            best = max(population.individuals, key=lambda x: x.fitness)
            save_genome(best.genome, "interrupted_best_genome.pkl")
        return None
    except Exception as e:
        return None

def load_best_genome(filename="best_genome.pkl") -> Optional[Genome]:
    try:
        filepath = os.path.join("models", filename)
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, "rb") as f:
            genome = pickle.load(f)
        
        return genome
    except Exception as e:
        return None

def test_best_genome(headless: bool = False, filename="best_genome.pkl") -> Optional[Genome]:
    genome = load_best_genome(filename)
    if genome:
        wrapper = NeatGameWrapper(headless=headless, show_zones=True)  # Enable zones for testing
        wrapper.run_visual_test(genome)
        return genome