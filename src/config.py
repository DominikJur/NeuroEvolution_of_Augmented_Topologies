# Optimized config.py for spike avoidance game

class Config:
    def __init__(self):
        self.population_size = 150
        self.survival_threshold = 0.15  # Keep aggressive selection for precision timing
        
        # CRITICAL FIX: Much lower compatibility threshold to force more species
        self.compatibility_threshold = 2.5 
        
        # OPTIMIZED: Faster species innovation for shorter training runs
        self.stagnation_threshold = 15  # Reduced from 25 for faster innovation
        
        # Keep the rest of your stable parameters
        self.weight_init_stdev = 1.0
        self.weight_min = -8.0
        self.weight_max = 8.0
        self.weight_mutation_rate = 0.4
        self.weight_mutate_power = 0.6
        self.weight_replace_rate = 0.15
        
        self.bias_init_stdev = 0.8
        self.bias_min = -4.0
        self.bias_max = 4.0
        self.bias_mutation_rate = 0.4
        self.bias_mutate_power = 0.4
        self.bias_replace_rate = 0.1
        
        self.add_connection_rate = 0.5
        self.add_node_rate = 0.1
        self.remove_connection_rate = 0.1
        self.remove_node_rate = 0.05
        self.toggle_connection_rate = 0.1

config = Config()
