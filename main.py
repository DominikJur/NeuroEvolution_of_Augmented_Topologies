import random
import sys
import pygame
from pygame import mixer

from src.coin_class import Coin
from src.player_class import Player
from src.spike_classes import (Ceilling_Spike, East_Wall_Spike, 
                               Floor_Spike, West_Wall_Spike)


class Game:
    def __init__(self):
        pygame.init()
        
        # Game constants
        self.scale = 2
        self.user_x = 288 * self.scale
        self.user_y = 352 * self.scale
        
        # Initialize display
        self.screen = pygame.display.set_mode((self.user_x, self.user_y))
        pygame.display.set_caption("Beware of the skewers!")
        
        # Load persistent data
        self.high_score = self._load_high_score()
        self.start_high_score = self.high_score
        self.coin_total = self._load_coin_total()
        
        # Game state
        self.running = False
        self.score = 0
        
        # Initialize game objects
        self.player = Player(self.scale, self.user_x, self.user_y)
        self.east_spikes = []
        self.west_spikes = []
        self.coin_list = []
        
        # Initialize sounds
        self.coin_sound = mixer.Sound("aduio/coin_collect.mp3")
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        
        # Initialize static elements
        self._setup_floor_and_ceiling()
    
    def _load_high_score(self):
        """Load high score from file"""
        try:
            with open("data/high_score.txt") as file:
                return file.read().strip()
        except FileNotFoundError:
            return "0"
    
    def _load_coin_total(self):
        """Load total coins from file"""
        try:
            with open("data/total_coins.txt") as file:
                return int(file.read().strip())
        except FileNotFoundError:
            return 0
    
    def _save_data(self):
        """Save high score and coin total to files"""
        if int(self.start_high_score) < int(self.high_score):
            with open("data/high_score.txt", "w") as file:
                file.write(f"{self.high_score}")
        with open("data/total_coins.txt", "w") as file:
            file.write(f"{self.coin_total}")
    
    def _setup_floor_and_ceiling(self):
        """Initialize floor and ceiling spikes"""
        self.floor_rect = pygame.Rect(0, round(self.user_y * 0.9), self.user_x, self.user_y)
        self.floor_spikes = [Floor_Spike(self.scale) for _ in range(9)]
        self.floor_spikes = [
            spike.set_position(i * self.scale * 32, round(self.user_y * 0.9) - spike.rect.height) 
            for i, spike in enumerate(self.floor_spikes)
        ]
        
        self.ceiling_rect = pygame.Rect(0, -self.user_y, self.user_x, self.user_y + round(self.user_y * 0.1))
        self.ceiling_spikes = [Ceilling_Spike(self.scale) for _ in range(9)]
        self.ceiling_spikes = [
            spike.set_position(i * self.scale * 32, round(self.user_y * 0.1)) 
            for i, spike in enumerate(self.ceiling_spikes)
        ]
    
    def draw_text(self, text, size, x, y):
        """Draw text on the screen"""
        font = pygame.font.Font("font\\Pixeltype.ttf", size)
        text_surface = font.render(text, True, "white")
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(text_surface, text_rect)
    
    def update_high_score(self):
        """Update high score if current score is higher"""
        if int(self.high_score) < int(self.score):
            self.high_score = str(self.score)
    
    def _randomize_spikes(self, flag):
        """Generate random spikes based on collision flag and score"""
        if flag == 0:
            return [], [], 0
        
        # Determine spike count based on score
        if self.score < 5:
            count = 2
        elif self.score < 10:
            count = 3
        elif self.score < 20:
            count = 4
        else:
            count = 5
        
        if flag == 1:
            return [East_Wall_Spike(self.scale) for _ in range(count)], [], count
        else:
            return [], [West_Wall_Spike(self.scale) for _ in range(count)], count
    
    def generate_spikes(self):
        """Generate wall spikes when player collides with wall"""
        east_spikes, west_spikes, count = self._randomize_spikes(self.player.check_wall_collision())
        pos_list = random.sample(range(2, 9), count)
        
        west_spikes = [
            spike.set_position(0, pos_list.pop() * 32 * self.scale) 
            for spike in west_spikes
        ]
        east_spikes = [
            spike.set_position(self.user_x - spike.rect.width, pos_list.pop() * 32 * self.scale) 
            for spike in east_spikes
        ]
        
        return east_spikes, west_spikes
    
    def manage_spikes(self):
        """Handle spike rendering and collision detection"""
        for spike in self.east_spikes + self.west_spikes:
            self.screen.blit(spike.image, spike.rect)
            if spike.rect.colliderect(self.player.rect):
                self.player.death()
    
    def generate_coins(self):
        """Generate a new coin if none exist"""
        if not self.coin_list:
            coin = Coin(self.scale)
            x = random.randint(round(0.2 * self.user_x), round(0.8 * self.user_x))
            y = random.randint(round(0.2 * self.user_y), round(0.8 * self.user_y))
            self.coin_list = [coin.set_position(x, y)]
    
    def manage_coins(self):
        """Handle coin rendering and collection"""
        for coin in self.coin_list:
            self.screen.blit(coin.image, coin.rect)
            if coin.rect.colliderect(self.player.rect):
                self.coin_sound.play()
                self.coin_list = []
                self.coin_total += 1
    
    def check_static_spike_collisions(self):
        """Check collisions with floor and ceiling spikes"""
        for spike in self.floor_spikes + self.ceiling_spikes:
            if spike.rect.colliderect(self.player.rect):
                self.player.death()
    
    def handle_wall_collision(self):
        """Handle player collision with walls"""
        if self.player.check_wall_collision() != 0:
            self.player.after_collision()
            self.score += 1
            self.generate_coins()
            self.east_spikes, self.west_spikes = self.generate_spikes()
    
    def reset_game(self):
        """Reset game state for new game"""
        self.player.defult_pos()
        self.player.dead = False
        self.coin_list = []
        self.score = 0
        self.east_spikes = []
        self.west_spikes = []
        self.running = False
    
    def handle_events(self):
        """Handle all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._save_data()
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if self.player.dead:
                    self.reset_game()
                elif not self.running:
                    self.running = True
                    self.player.jump()
                else:
                    self.player.jump()
            
            if self.player.dead and event.type == pygame.MOUSEBUTTONDOWN:
                # Check if clicked on retry button
                button_x = self.user_x // 3
                button_y = self.user_y // 5 * 2
                button_width = self.user_x // 3
                button_height = self.user_y // 6
                
                if (button_x < event.pos[0] < button_x + button_width and 
                    button_y < event.pos[1] < button_y + button_height):
                    self.reset_game()
    
    def draw_background(self):
        """Draw the game background"""
        self.screen.fill("#c3c3c3")
        
        # Draw floor
        pygame.draw.rect(self.screen, "#5c5c5c", self.floor_rect)
        for spike in self.floor_spikes:
            self.screen.blit(spike.image, spike.rect)
        
        # Draw ceiling
        pygame.draw.rect(self.screen, "#5c5c5c", self.ceiling_rect)
        for spike in self.ceiling_spikes:
            self.screen.blit(spike.image, spike.rect)
    
    def draw_ui(self):
        """Draw the user interface elements"""
        self.draw_text(f"Score: {self.score}", 40 * self.scale, 
                      self.user_x // 2, self.user_y // 3 - 12 * self.scale)
        self.draw_text(f"High Score: {self.high_score}", 40 * self.scale, 
                      self.user_x // 2, self.user_y // 5 - 12 * self.scale)
        self.draw_text(f"Coins: {self.coin_total}", 40 * self.scale, 
                      self.user_x // 2, self.user_y // 4 * 3)
    
    def draw_start_screen(self):
        """Draw the start screen"""
        self.player.defult_pos()
        self.player.start_screen_animation()
        self.draw_text("Press SPACE to start", 40 * self.scale, 
                      self.user_x // 2, self.user_y // 2 + 30 * self.scale)
    
    def draw_death_screen(self):
        """Draw the death/retry screen"""
        # Draw retry button
        button_rect = pygame.Rect(self.user_x // 3, self.user_y // 5 * 2, 
                                 self.user_x // 3, self.user_y // 6)
        pygame.draw.rect(self.screen, "#fd5f5f", button_rect, 90, 30)
        pygame.draw.rect(self.screen, "#bf5959", button_rect, 10, 30)
        self.draw_text("Retry", 40 * self.scale, 
                      self.user_x // 2, self.user_y // 2 - 16 * self.scale)
    
    def update_game(self):
        """Update game logic during active gameplay"""
        self.check_static_spike_collisions()
        self.handle_wall_collision()
        self.manage_coins()
        self.manage_spikes()
        self.player.update()
        
        if self.player.dead:
            self.west_spikes = []
            self.east_spikes = []
            self.running = False
    
    def run(self):
        """Main game loop"""
        while True:
            self.handle_events()
            self.draw_background()
            self.draw_ui()
            
            if not self.player.dead:
                if self.running:
                    self.update_game()
                else:
                    self.draw_start_screen()
            else:
                self.draw_death_screen()
            
            # Draw player
            self.screen.blit(self.player.image, self.player.rect)
            
            # Update high score
            self.update_high_score()
            
            # Update display
            pygame.display.update()
            self.clock.tick(30)


def main():
    """Entry point of the game"""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()