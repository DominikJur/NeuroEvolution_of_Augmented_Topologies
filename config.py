# NEAT Configuration for Don't Touch the Spikes
# Adjust these values to change how NEAT evolves

class GameConfig:
    """Game-specific settings"""
    # How long each game runs (in frames)
    MAX_GAME_TIME = 1000
    
    # Fitness rewards
    SURVIVAL_REWARD = 1      # Points per frame survived
    SCORE_REWARD = 500       # Points per game score point
    SCORE_BONUS = 100        # Bonus when score increases
    
    # Input normalization
    MAX_VELOCITY = 8         # Expected max player velocity
    MAX_GRAVITY = 10         # Expected max gravity
    
class NEATConfig:
    """NEAT algorithm settings"""
    # Population
    POPULATION_SIZE = 50
    SURVIVAL_RATE = 0.1      # Top 20% survive to next generation
    
    # Network structure
    INPUT_SIZE = 6           # Don't change unless you modify inputs
    OUTPUT_SIZE = 1          # Don't change unless you modify outputs
    
    # Mutation rates
    ADD_CONNECTION_RATE = 0.5
    ADD_NODE_RATE = 0.2
    REMOVE_CONNECTION_RATE = 0.1
    REMOVE_NODE_RATE = 0.05
    
    # Weight mutation
    WEIGHT_MUTATION_RATE = 0.8
    WEIGHT_MUTATE_POWER = 0.5
    WEIGHT_REPLACE_RATE = 0.1
    WEIGHT_MAX = 20.0
    WEIGHT_MIN = -20.0
    
    # Bias mutation  
    BIAS_MUTATION_RATE = 0.7
    BIAS_MUTATE_POWER = 0.5
    BIAS_REPLACE_RATE = 0.1
    BIAS_MAX = 20.0
    BIAS_MIN = -20.0

# Preset configurations for different scenarios
PRESETS = {
    'quick': {
        'population': 20,
        'generations': 10,
        'max_time': 500
    },
    'balanced': {
        'population': 50, 
        'generations': 50,
        'max_time': 1000
    },
    'thorough': {
        'population': 100,
        'generations': 100, 
        'max_time': 2000
    },
    'test': {
        'population': 10,
        'generations': 5,
        'max_time': 200
    }
}