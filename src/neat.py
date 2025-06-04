import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import copy
import math

# Global innovation tracker
class InnovationTracker:
    def __init__(self):
        self.counter = 0
        self.innovations = {}  # (from_node, to_node) -> innovation_number
    
    def get_innovation(self, from_node, to_node):
        key = (from_node, to_node)
        if key not in self.innovations:
            self.innovations[key] = self.counter
            self.counter += 1
        return self.innovations[key]

innovation_tracker = InnovationTracker()

# Global genome ID counter
genome_id_counter = 0

def next_genome_id():
    global genome_id_counter
    genome_id_counter += 1
    return genome_id_counter

class NeuronGene:
    def __init__(self, node_id, neuron_type="hidden", bias=0.0):
        self.node_id = node_id
        self.type = neuron_type  # "input", "output", "hidden"
        self.bias = bias
        self.activation = "tanh" 

class ConnectionID:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
    
    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node
    
    def __hash__(self):
        return hash((self.from_node, self.to_node))

class ConnectionGene:
    def __init__(self, from_node, to_node, weight=0.0, enabled=True, innovation=None):
        self.id = ConnectionID(from_node, to_node)
        self.weight = weight
        self.enabled = enabled
        if innovation is None:
            self.innovation = innovation_tracker.get_innovation(from_node, to_node)
        else:
            self.innovation = innovation
 
class Genome:
    def __init__(self, genome_id, input_size, output_size):
        self.id = genome_id
        self.nodes = {}  # node_id -> NeuronGene
        self.connections = {}  # innovation -> ConnectionGene
        self.node_counter = 0
        self.input_size = input_size
        self.output_size = output_size
        self.fitness = 0.0

        # Create input nodes (negative IDs)
        for i in range(input_size):
            node_id = -(i + 1)
            self.nodes[node_id] = NeuronGene(node_id, "input", bias=0.0)

        # Create output nodes (positive IDs starting from 0)
        for i in range(output_size):
            self.nodes[i] = NeuronGene(i, "output", bias=np.random.randn())
            self.node_counter = max(self.node_counter, i + 1)

    def add_node(self, neuron_gene=None, bias=None):
        if neuron_gene is None:
            node_id = self.node_counter
            self.node_counter += 1
            neuron_gene = NeuronGene(node_id, "hidden", bias if bias is not None else np.random.randn())
        
        self.nodes[neuron_gene.node_id] = neuron_gene
        return neuron_gene.node_id

    def add_connection(self, from_node, to_node, weight=0.0, enabled=True):
        if self.has_connection(from_node, to_node):
            return False
        
        innovation = innovation_tracker.get_innovation(from_node, to_node)
        connection = ConnectionGene(from_node, to_node, weight, enabled, innovation)
        self.connections[innovation] = connection
        return True

    def has_connection(self, from_node, to_node):
        for conn in self.connections.values():
            if conn.id.from_node == from_node and conn.id.to_node == to_node:
                return True
        return False

    def find_connection(self, from_node, to_node):
        for conn in self.connections.values():
            if conn.id.from_node == from_node and conn.id.to_node == to_node:
                return conn
        return None

    def remove_connection(self, from_node, to_node):
        to_remove = None
        for innovation, conn in self.connections.items():
            if conn.id.from_node == from_node and conn.id.to_node == to_node:
                to_remove = innovation
                break
        if to_remove is not None:
            del self.connections[to_remove]

    def remove_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]

    def remove_connections_by_node(self, node_id):
        to_remove = []
        for innovation, conn in self.connections.items():
            if conn.id.from_node == node_id or conn.id.to_node == node_id:
                to_remove.append(innovation)
        for innovation in to_remove:
            del self.connections[innovation]

    def find_neuron(self, node_id):
        return self.nodes.get(node_id)

    def num_inputs(self):
        return self.input_size

    def num_outputs(self):
        return self.output_size

    def get_input_nodes(self):
        return [n for n in self.nodes.values() if n.type == "input"]

    def get_output_nodes(self):
        return [n for n in self.nodes.values() if n.type == "output"]

    def get_hidden_nodes(self):
        return [n for n in self.nodes.values() if n.type == "hidden"]

class Individual:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.fitness = 0.0
          
def crossover_neuron(dominant, recessive):
    """Crossover between two neuron genes"""
    assert dominant.node_id == recessive.node_id, "Neuron IDs must match for crossover"
    bias = dominant.bias if np.random.rand() < 0.5 else recessive.bias
    activation = dominant.activation if np.random.rand() < 0.5 else recessive.activation
    
    result = NeuronGene(dominant.node_id, dominant.type, bias)
    result.activation = activation
    return result
 
def crossover_connection(conn1, conn2):
    """Crossover between two connection genes"""
    if np.random.rand() < 0.5:
        return ConnectionGene(conn1.id.from_node, conn1.id.to_node, conn1.weight, conn1.enabled, conn1.innovation)
    else:
        return ConnectionGene(conn2.id.from_node, conn2.id.to_node, conn2.weight, conn2.enabled, conn2.innovation)

def genome_crossover(dominant_ind, recessive_ind):
    """Crossover two genomes, keeping structural genes from dominant parent"""
    dominant = dominant_ind.genome
    recessive = recessive_ind.genome
    
    offspring = Genome(
        next_genome_id(),
        dominant.num_inputs(),
        dominant.num_outputs()
    )
    
    # Inherit neurons from both parents
    all_node_ids = set(dominant.nodes.keys()) | set(recessive.nodes.keys())
    for node_id in all_node_ids:
        dominant_neuron = dominant.find_neuron(node_id)
        recessive_neuron = recessive.find_neuron(node_id)
        
        if dominant_neuron and recessive_neuron:
            # Both have this neuron - crossover
            offspring.nodes[node_id] = crossover_neuron(dominant_neuron, recessive_neuron)
        elif dominant_neuron:
            # Only dominant has it - inherit from dominant
            offspring.nodes[node_id] = copy.deepcopy(dominant_neuron)
        # Don't inherit excess genes from recessive parent

    # Inherit connections
    all_innovations = set(dominant.connections.keys()) | set(recessive.connections.keys())
    for innovation in all_innovations:
        dominant_conn = dominant.connections.get(innovation)
        recessive_conn = recessive.connections.get(innovation)
        
        if dominant_conn and recessive_conn:
            # Both have this connection - crossover
            offspring.connections[innovation] = crossover_connection(dominant_conn, recessive_conn)
        elif dominant_conn:
            # Only dominant has it - inherit from dominant
            offspring.connections[innovation] = copy.deepcopy(dominant_conn)
        # Don't inherit excess genes from recessive parent

    return Individual(offspring)

def mutate_add_connection(genome: Genome, max_attempts=50):
    """Add a new connection between two existing nodes"""
    input_nodes = genome.get_input_nodes()
    output_nodes = genome.get_output_nodes()
    hidden_nodes = genome.get_hidden_nodes()
    
    all_nodes = input_nodes + output_nodes + hidden_nodes
    
    if len(all_nodes) < 2:
        return False

    for _ in range(max_attempts):
        from_node = np.random.choice(all_nodes)
        to_node = np.random.choice(all_nodes)

        # Can't connect to input nodes or from output nodes
        if to_node.type == "input" or from_node.type == "output":
            continue
            
        # Can't connect to self
        if from_node.node_id == to_node.node_id:
            continue
            
        # Check if connection already exists
        if genome.has_connection(from_node.node_id, to_node.node_id):
            continue

        weight = np.random.randn()
        return genome.add_connection(from_node.node_id, to_node.node_id, weight)
    
    return False

def mutate_remove_connection(genome: Genome):
    """Remove a random connection"""
    if not genome.connections:
        return False

    connections = list(genome.connections.values())
    connection = np.random.choice(connections)
    genome.remove_connection(connection.id.from_node, connection.id.to_node)
    return True

def mutate_add_neuron(genome: Genome):
    """Add a new hidden neuron by splitting an existing connection"""
    enabled_connections = [conn for conn in genome.connections.values() if conn.enabled]
    if not enabled_connections:
        return False

    connection = np.random.choice(enabled_connections)
    
    # Disable the original connection
    connection.enabled = False

    # Create a new neuron
    new_node_id = genome.add_node(bias=np.random.randn())

    # Create two new connections
    genome.add_connection(connection.id.from_node, new_node_id, weight=1.0)
    genome.add_connection(new_node_id, connection.id.to_node, weight=connection.weight)
    return True

def mutate_remove_neuron(genome: Genome):
    """Remove a random hidden neuron and its connections"""
    hidden_nodes = genome.get_hidden_nodes()
    if not hidden_nodes:
        return False

    neuron = np.random.choice(hidden_nodes)
    genome.remove_node(neuron.node_id)
    genome.remove_connections_by_node(neuron.node_id)
    return True

def mutate_weight(genome: Genome, config):
    """Mutate connection weights"""
    for connection in genome.connections.values():
        if np.random.rand() < config.weight_mutation_rate:
            if np.random.rand() < config.weight_replace_rate:
                connection.weight = np.random.randn() * config.weight_init_stdev
            else:
                connection.weight += np.random.randn() * config.weight_mutate_power
                connection.weight = np.clip(connection.weight, config.weight_min, config.weight_max)

def mutate_bias(genome: Genome, config):
    """Mutate neuron biases"""
    for neuron in genome.nodes.values():
        if neuron.type != "input" and np.random.rand() < config.bias_mutation_rate:
            if np.random.rand() < config.bias_replace_rate:
                neuron.bias = np.random.randn() * config.bias_init_stdev
            else:
                neuron.bias += np.random.randn() * config.bias_mutate_power
                neuron.bias = np.clip(neuron.bias, config.bias_min, config.bias_max)

class Config:
    def __init__(self):
        # Population parameters
        self.population_size = 150
        self.survival_threshold = 0.2
        
        # Weight parameters
        self.weight_init_stdev = 1.0
        self.weight_min = -20.0
        self.weight_max = 20.0
        self.weight_mutation_rate = 0.8
        self.weight_mutate_power = 0.5
        self.weight_replace_rate = 0.1
        
        # Bias parameters
        self.bias_init_stdev = 1.0
        self.bias_min = -20.0
        self.bias_max = 20.0
        self.bias_mutation_rate = 0.7
        self.bias_mutate_power = 0.5
        self.bias_replace_rate = 0.1
        
        # Structural mutation rates
        self.add_connection_rate = 0.5
        self.add_node_rate = 0.2
        self.remove_connection_rate = 0.1
        self.remove_node_rate = 0.05

config = Config()

class Population:
    def __init__(self, config, input_size, output_size):
        self.config = config
        self.individuals = []
        self.input_size = input_size
        self.output_size = output_size
        self.generation = 0
        
        # Initialize population
        for i in range(config.population_size):
            genome = self.create_initial_genome()
            individual = Individual(genome)
            self.individuals.append(individual)
    
    def create_initial_genome(self):
        """Create a minimal genome with direct input-output connections"""
        genome = Genome(next_genome_id(), self.input_size, self.output_size)
        
        # Connect all inputs to all outputs
        for i in range(self.input_size):
            input_id = -(i + 1)
            for j in range(self.output_size):
                output_id = j
                weight = np.random.randn()
                genome.add_connection(input_id, output_id, weight)
        
        return genome

    def run(self, fitness_function, num_generations):
        """Run the genetic algorithm for the specified number of generations."""
        for generation in range(num_generations):
            self.generation = generation
            
            # Evaluate fitness
            for individual in self.individuals:
                individual.fitness = fitness_function(individual.genome)
            
            # Sort by fitness (higher is better)
            self.individuals.sort(key=lambda x: x.fitness, reverse=True)
            
            best_fitness = self.individuals[0].fitness
            print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            # Create next generation
            if generation < num_generations - 1:
                self.individuals = self.reproduce()
        
        return self.individuals[0]  # Return best individual

    def reproduce(self):
        """Create a new generation through selection, crossover, and mutation."""
        # Keep top performers
        cutoff = int(self.config.survival_threshold * len(self.individuals))
        survivors = self.individuals[:cutoff]
        
        new_generation = []
        
        while len(new_generation) < self.config.population_size:
            # Select parents
            parent1 = np.random.choice(survivors)
            parent2 = np.random.choice(survivors)
            
            # Determine which parent is more fit
            if parent1.fitness >= parent2.fitness:
                dominant, recessive = parent1, parent2
            else:
                dominant, recessive = parent2, parent1
            
            # Crossover
            offspring = genome_crossover(dominant, recessive)
            
            # Mutation
            self.mutate(offspring.genome)
            
            new_generation.append(offspring)
        
        return new_generation

    def mutate(self, genome):
        """Apply various mutations to a genome"""
        # Structural mutations
        if np.random.rand() < self.config.add_connection_rate:
            mutate_add_connection(genome)
        
        if np.random.rand() < self.config.add_node_rate:
            mutate_add_neuron(genome)
        
        if np.random.rand() < self.config.remove_connection_rate:
            mutate_remove_connection(genome)
        
        if np.random.rand() < self.config.remove_node_rate:
            mutate_remove_neuron(genome)
        
        # Parameter mutations
        mutate_weight(genome, self.config)
        mutate_bias(genome, self.config)

class NeuronInput:
    def __init__(self, node_id, weight=1.0):
        self.node_id = node_id
        self.weight = weight

class Neuron:
    def __init__(self, neuron_gene, inputs):
        self.node_id = neuron_gene.node_id
        self.inputs = inputs
        self.bias = neuron_gene.bias
        self.activation = neuron_gene.activation
        self.type = neuron_gene.type

class FeedForwardNetwork:
    def __init__(self, input_ids, output_ids, neurons):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.neurons = neurons
    
    def activate(self, inputs):
        """Activate the network with the given inputs."""
        assert len(inputs) == len(self.input_ids), f"Input size mismatch: expected {len(self.input_ids)}, got {len(inputs)}"
        
        # Initialize all neuron outputs
        neuron_outputs = {}
        
        # Set input values
        for i, input_value in enumerate(inputs):
            neuron_outputs[self.input_ids[i]] = input_value
        
        # Process neurons in topological order
        processed = set(self.input_ids)
        
        while len(processed) < len(self.neurons) + len(self.input_ids):
            made_progress = False
            
            for neuron in self.neurons:
                if neuron.node_id in processed:
                    continue
                
                # Check if all inputs are ready
                inputs_ready = all(inp.node_id in neuron_outputs for inp in neuron.inputs)
                
                if inputs_ready:
                    # Calculate neuron output
                    value = sum(neuron_outputs[inp.node_id] * inp.weight for inp in neuron.inputs)
                    value += neuron.bias

                    if neuron.activation == "tanh":
                        neuron_outputs[neuron.node_id] = np.tanh(value)
                    elif neuron.activation == "sigmoid":
                        neuron_outputs[neuron.node_id] = 1 / (1 + np.exp(-np.clip(value, -500, 500)))
                    elif neuron.activation == "relu":
                        neuron_outputs[neuron.node_id] = max(0, value)
                    else:
                        neuron_outputs[neuron.node_id] = np.tanh(value)  # Default to tanh

                    processed.add(neuron.node_id)
                    made_progress = True
            
            if not made_progress:
                break  # Avoid infinite loop if there are cycles

        # Collect output values
        outputs = []
        for output_id in self.output_ids:
            if output_id in neuron_outputs:
                outputs.append(neuron_outputs[output_id])
            else:
                outputs.append(0.0)  # Default value if neuron not processed
        
        return outputs
    
    @staticmethod
    def create_from_genome(genome):
        """Create a FeedForwardNetwork from a genome."""
        # Get input and output node IDs
        input_ids = [node.node_id for node in genome.get_input_nodes()]
        output_ids = [node.node_id for node in genome.get_output_nodes()]
        
        # Create neurons for non-input nodes
        neurons = []
        for node in genome.nodes.values():
            if node.type != "input":
                # Find all incoming connections
                inputs = []
                for conn in genome.connections.values():
                    if conn.id.to_node == node.node_id and conn.enabled:
                        inputs.append(NeuronInput(conn.id.from_node, conn.weight))
                
                if inputs:  # Only create neuron if it has inputs
                    neurons.append(Neuron(node, inputs))
        
        return FeedForwardNetwork(input_ids, output_ids, neurons)

def xor_fitness(genome):
    """Fitness function for XOR problem"""
    network = FeedForwardNetwork.create_from_genome(genome)
    
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_outputs = [0, 1, 1, 0]
    
    total_error = 0.0
    for inputs, expected in zip(xor_inputs, xor_outputs):
        output = network.activate(inputs)
        error = (output[0] - expected) ** 2
        total_error += error
    
    # Return fitness (higher is better)
    return 4.0 - total_error

def main():
    """Main function to run NEAT on XOR problem"""
    CONFIG = Config()
    num_generations = 1000
    
    population = Population(CONFIG, input_size=2, output_size=1)
    winner = population.run(xor_fitness, num_generations)
    
    print(f"\nBest genome (ID: {winner.genome.id}) with fitness: {winner.fitness:.4f}")
    print(f"Nodes: {len(winner.genome.nodes)}")
    print(f"Connections: {len(winner.genome.connections)}")
    
    # Test the winner
    network = FeedForwardNetwork.create_from_genome(winner.genome)
    print("\nTesting winner on XOR:")
    for inputs in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        output = network.activate(inputs)
        print(f"Input: {inputs}, Output: {output[0]:.4f}")
    
    return winner

if __name__ == "__main__":
    main()