class Config:
    def __init__(self):
        self.population_size = 150
        self.survival_threshold = 0.15
        
        # SOLUTION 1: Lower compatibility threshold
        self.compatibility_threshold = 2.0  # Was 4.5 - much more sensitive
        
        self.stagnation_threshold = 20
        
        # SOLUTION 2: Increase initial weight variance
        self.weight_init_stdev = 1.5  # Was 0.5 - more diverse weights
        self.weight_min = -8.0
        self.weight_max = 8.0
        self.weight_mutation_rate = 0.8
        self.weight_mutate_power = 0.8  # Was 0.4 - larger mutations
        self.weight_replace_rate = 0.2  # Was 0.1 - more radical changes
        
        # SOLUTION 3: Increase bias variance  
        self.bias_init_stdev = 1.0  # Was 0.3 - more diverse biases
        self.bias_min = -4.0
        self.bias_max = 4.0
        self.bias_mutation_rate = 0.6
        self.bias_mutate_power = 0.5  # Was 0.2 - larger mutations
        self.bias_replace_rate = 0.15  # Was 0.08 - more replacement
        
        # SOLUTION 4: Increase structural mutation rates early
        self.add_connection_rate = 0.7  # Was 0.5 - more connections
        self.add_node_rate = 0.3  # Was 0.2 - more complexity
        self.remove_connection_rate = 0.05  # Was 0.0 - some pruning
        self.remove_node_rate = 0.02  # Was 0.0 - some simplification
        self.toggle_connection_rate = 0.05  # Was 0.02 - more toggling

config = Config()
