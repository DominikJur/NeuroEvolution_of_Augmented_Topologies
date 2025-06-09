import copy
import math
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config


class InnovationTracker:
    def __init__(self):
        self.counter = 0
        self.innovations = {}
    
    def get_innovation(self, from_node, to_node):
        key = (from_node, to_node)
        if key not in self.innovations:
            self.innovations[key] = self.counter
            self.counter += 1
        return self.innovations[key]

innovation_tracker = InnovationTracker()

genome_id_counter = 0

def next_genome_id():
    global genome_id_counter
    genome_id_counter += 1
    return genome_id_counter

class NeuronGene:
    def __init__(self, node_id, neuron_type="hidden", bias=0.0):
        self.node_id = node_id
        self.type = neuron_type
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
        self.nodes = {}
        self.connections = {}
        self.node_counter = 0
        self.input_size = input_size
        self.output_size = output_size
        self.fitness = 0.0

        for i in range(input_size):
            node_id = -(i + 1)
            self.nodes[node_id] = NeuronGene(node_id, "input", bias=0.0)

        for i in range(output_size):
            self.nodes[i] = NeuronGene(i, "output", bias=np.random.randn() * 0.5)
            self.node_counter = max(self.node_counter, i + 1)

    def add_node(self, neuron_gene=None, bias=None):
        if neuron_gene is None:
            node_id = self.node_counter
            self.node_counter += 1
            neuron_gene = NeuronGene(node_id, "hidden", bias if bias is not None else np.random.randn() * 0.5)
        
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

    def compatibility_distance(self, other):
        c1, c2, c3 = 1.0, 1.0, 0.4
        
        my_innovations = set(self.connections.keys())
        other_innovations = set(other.connections.keys())
        
        matching = my_innovations & other_innovations
        disjoint = my_innovations ^ other_innovations
        
        if matching:
            max_innovation = max(max(my_innovations), max(other_innovations))
            excess = sum(1 for innov in disjoint if innov > max(matching))
            disjoint_count = len(disjoint) - excess
        else:
            excess = len(disjoint)
            disjoint_count = 0
        
        weight_diff = 0.0
        if matching:
            for innov in matching:
                weight_diff += abs(self.connections[innov].weight - other.connections[innov].weight)
            weight_diff /= len(matching)
        
        n = max(len(self.connections), len(other.connections), 1)
        
        distance = (c1 * excess / n) + (c2 * disjoint_count / n) + (c3 * weight_diff)
        return distance

class Individual:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.fitness = 0.0
        self.species_id = None
          
def crossover_neuron(dominant, recessive):
    assert dominant.node_id == recessive.node_id
    bias = dominant.bias if np.random.rand() < 0.7 else recessive.bias
    activation = dominant.activation if np.random.rand() < 0.7 else recessive.activation
    
    result = NeuronGene(dominant.node_id, dominant.type, bias)
    result.activation = activation
    return result
 
def crossover_connection(conn1, conn2):
    if np.random.rand() < 0.7:
        return ConnectionGene(conn1.id.from_node, conn1.id.to_node, conn1.weight, conn1.enabled, conn1.innovation)
    else:
        return ConnectionGene(conn2.id.from_node, conn2.id.to_node, conn2.weight, conn2.enabled, conn2.innovation)

def genome_crossover(dominant_ind, recessive_ind):
    dominant = dominant_ind.genome
    recessive = recessive_ind.genome
    
    offspring = Genome(
        next_genome_id(),
        dominant.num_inputs(),
        dominant.num_outputs()
    )
    
    all_node_ids = set(dominant.nodes.keys()) | set(recessive.nodes.keys())
    for node_id in all_node_ids:
        dominant_neuron = dominant.find_neuron(node_id)
        recessive_neuron = recessive.find_neuron(node_id)
        
        if dominant_neuron and recessive_neuron:
            offspring.nodes[node_id] = crossover_neuron(dominant_neuron, recessive_neuron)
        elif dominant_neuron:
            offspring.nodes[node_id] = copy.deepcopy(dominant_neuron)

    all_innovations = set(dominant.connections.keys()) | set(recessive.connections.keys())
    for innovation in all_innovations:
        dominant_conn = dominant.connections.get(innovation)
        recessive_conn = recessive.connections.get(innovation)
        
        if dominant_conn and recessive_conn:
            offspring.connections[innovation] = crossover_connection(dominant_conn, recessive_conn)
         
                
        elif dominant_conn:
            offspring.connections[innovation] = copy.deepcopy(dominant_conn)

    return Individual(offspring)

def mutate_add_connection(genome: Genome, max_attempts=50):
    input_nodes = genome.get_input_nodes()
    output_nodes = genome.get_output_nodes()
    hidden_nodes = genome.get_hidden_nodes()
    
    all_nodes = input_nodes + output_nodes + hidden_nodes
    
    if len(all_nodes) < 2:
        return False

    for _ in range(max_attempts):
        from_node = np.random.choice(all_nodes)
        to_node = np.random.choice(all_nodes)

        if to_node.type == "input" or from_node.type == "output":
            continue
            
        if from_node.node_id == to_node.node_id:
            continue
            
        if genome.has_connection(from_node.node_id, to_node.node_id):
            continue

        weight = np.random.randn() * 0.5
        return genome.add_connection(from_node.node_id, to_node.node_id, weight)
    
    return False

def mutate_remove_connection(genome: Genome):
    if not genome.connections:
        return False

    connections = list(genome.connections.values())
    connection = np.random.choice(connections)
    genome.remove_connection(connection.id.from_node, connection.id.to_node)
    return True

def mutate_add_neuron(genome: Genome):
    enabled_connections = [conn for conn in genome.connections.values() if conn.enabled]
    if not enabled_connections:
        return False

    connection = np.random.choice(enabled_connections)
    connection.enabled = False

    new_node_id = genome.add_node(bias=np.random.randn() * 0.3)

    # Add the split connections
    genome.add_connection(connection.id.from_node, new_node_id, weight=1.0)
    genome.add_connection(new_node_id, connection.id.to_node, weight=connection.weight)
    
    # FORCE connection to output if the new node isn't already connected
    if not any(conn.id.from_node == new_node_id and conn.id.to_node == 0 
               for conn in genome.connections.values() if conn.enabled):
        genome.add_connection(new_node_id, 0, weight=np.random.normal(0, 0.3))
    
    return True

def mutate_remove_neuron(genome: Genome):
    hidden_nodes = genome.get_hidden_nodes()
    if not hidden_nodes:
        return False

    neuron = np.random.choice(hidden_nodes)
    genome.remove_node(neuron.node_id)
    genome.remove_connections_by_node(neuron.node_id)
    return True

def mutate_weight(genome: Genome, config):
    for connection in genome.connections.values():
        if np.random.rand() < config.weight_mutation_rate:
            if np.random.rand() < config.weight_replace_rate:
                connection.weight = np.random.randn() * config.weight_init_stdev
            else:
                connection.weight += np.random.randn() * config.weight_mutate_power
                connection.weight = np.clip(connection.weight, config.weight_min, config.weight_max)

def mutate_bias(genome: Genome, config):
    for neuron in genome.nodes.values():
        if neuron.type != "input" and np.random.rand() < config.bias_mutation_rate:
            if np.random.rand() < config.bias_replace_rate:
                neuron.bias = np.random.randn() * config.bias_init_stdev
            else:
                neuron.bias += np.random.randn() * config.bias_mutate_power
                neuron.bias = np.clip(neuron.bias, config.bias_min, config.bias_max)

def mutate_toggle_connection(genome: Genome):
    if not genome.connections:
        return False
    
    connection = np.random.choice(list(genome.connections.values()))
    connection.enabled = not connection.enabled
    return True

class Species:
    def __init__(self, species_id, representative):
        self.id = species_id
        self.representative = representative
        self.members = []
        self.max_fitness = 0.0
        self.avg_fitness = 0.0
        self.stagnation_count = 0
        
    def add_member(self, individual):
        individual.species_id = self.id
        self.members.append(individual)
        
    def update_fitness(self):
        if not self.members:
            self.avg_fitness = 0.0
            return
            
        fitnesses = [ind.fitness for ind in self.members]
        self.avg_fitness = np.mean(fitnesses)
        
        current_max = max(fitnesses)
        if current_max > self.max_fitness:
            self.max_fitness = current_max
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
            
    def get_adjusted_fitness(self):
        for member in self.members:
            member.adjusted_fitness = member.fitness / len(self.members)


class Population:
    def __init__(self, config, input_size, output_size):
        self.config = config
        self.individuals = []
        self.input_size = input_size
        self.output_size = output_size
        self.generation = 0
        self.species = []
        self.species_id_counter = 0
        
        for i in range(config.population_size):
            genome = self.create_initial_genome()
            individual = Individual(genome)
            self.individuals.append(individual)
    
    def create_initial_genome(self):
        genome = Genome(next_genome_id(), self.input_size, self.output_size)
        
        # SOLUTION 5: More diverse initial connectivity patterns
        connection_patterns = [
            0.3,  # Sparse networks
            0.5,  # Medium networks  
            0.7,  # Dense networks
            0.9   # Very dense networks
        ]
        
        # Randomly select connectivity pattern for this genome
        connection_prob = np.random.choice(connection_patterns)
        
        for i in range(self.input_size):
            input_id = -(i + 1)
            for j in range(self.output_size):
                output_id = j
                if np.random.rand() < connection_prob:
                    # SOLUTION 6: Much wider weight distribution
                    weight = np.random.normal(0, 2.0)  # Was 0.5
                    genome.add_connection(input_id, output_id, weight)
        
        # Force spike connections with VERY different weights per genome
        spike_weight_strategies = [
            lambda: np.random.normal(-3, 1),  # Strong avoidance
            lambda: np.random.normal(3, 1),   # Strong attraction (risky)
            lambda: np.random.normal(0, 0.5), # Neutral/learning
            lambda: np.random.uniform(-5, 5)  # Random strategy
        ]
        
        strategy = np.random.choice(spike_weight_strategies)
        
        # UPDATED: Force connections for new spike inputs (now -4 to -8 instead of -3 to -5)
        for spike_input in [-4, -5, -6, -7, -8]:  # New spike inputs
            if not genome.has_connection(spike_input, 0):
                weight = strategy()  # Apply selected strategy
                genome.add_connection(spike_input, 0, weight)
        
        # SOLUTION 7: Random initial hidden nodes for some genomes
        if np.random.rand() < 0.5:  # 50% get initial hidden nodes
            num_hidden = np.random.randint(1, 4)  # 1-3 hidden nodes
            
            for _ in range(num_hidden):
                hidden_id = genome.add_node(bias=np.random.normal(0, 1.0))
                
                # Random connectivity for hidden nodes
                input_id = np.random.choice(list(range(-self.input_size, 0)))
                output_id = np.random.choice(list(range(self.output_size)))
                
                genome.add_connection(input_id, hidden_id, np.random.normal(0, 1.5))
                genome.add_connection(hidden_id, output_id, np.random.normal(0, 1.5))
        
        return genome

    def speciate(self):
        for individual in self.individuals:
            individual.species_id = None
            
        for species in self.species:
            species.members = []
            
        for individual in self.individuals:
            placed = False
            for species in self.species:
                if individual.genome.compatibility_distance(species.representative.genome) < self.config.compatibility_threshold:
                    species.add_member(individual)
                    placed = True
                    break
                    
            if not placed:
                new_species = Species(self.species_id_counter, individual)
                self.species_id_counter += 1
                new_species.add_member(individual)
                self.species.append(new_species)
        
        self.species = [s for s in self.species if len(s.members) > 0]
        
        for species in self.species:
            species.update_fitness()
            species.get_adjusted_fitness()

    def run(self, fitness_function, num_generations):
        for generation in range(num_generations):
            self.generation = generation
            
            for individual in self.individuals:
                individual.fitness = fitness_function(individual.genome)
                individual.genome.fitness = individual.fitness
            
            self.speciate()
            
            self.individuals.sort(key=lambda x: x.fitness, reverse=True)
            
            best_fitness = self.individuals[0].fitness
            print(f"Generation {generation}: Best fitness = {best_fitness:.4f}, Species: {len(self.species)}")
            
            if generation < num_generations - 1:
                self.individuals = self.reproduce()
        
        return self.individuals[0]

    def reproduce(self):
        self.species = [s for s in self.species if s.stagnation_count < self.config.stagnation_threshold]
        
        if not self.species:
            self.species = [Species(0, self.individuals[0])]
            self.species[0].members = self.individuals[:10]
        
        total_adjusted_fitness = sum(s.avg_fitness for s in self.species if s.avg_fitness > 0)
        if total_adjusted_fitness == 0:
            offspring_counts = [self.config.population_size // len(self.species)] * len(self.species)
        else:
            offspring_counts = []
            for species in self.species:
                count = int((species.avg_fitness / total_adjusted_fitness) * self.config.population_size)
                offspring_counts.append(max(1, count))
        
        while sum(offspring_counts) < self.config.population_size:
            best_species_idx = max(range(len(self.species)), key=lambda i: self.species[i].avg_fitness)
            offspring_counts[best_species_idx] += 1
            
        while sum(offspring_counts) > self.config.population_size:
            worst_species_idx = min(range(len(self.species)), key=lambda i: self.species[i].avg_fitness)
            if offspring_counts[worst_species_idx] > 1:
                offspring_counts[worst_species_idx] -= 1
        
        new_generation = []
        
        for species, offspring_count in zip(self.species, offspring_counts):
            species.members.sort(key=lambda x: x.fitness, reverse=True)
            
            survivors_count = max(1, int(self.config.survival_threshold * len(species.members)))
            survivors = species.members[:survivors_count]
            
            if survivors:
                new_generation.append(survivors[0])
                offspring_count -= 1
            
            for _ in range(offspring_count):
                if len(survivors) == 1:
                    parent = survivors[0]
                    offspring = Individual(copy.deepcopy(parent.genome))
                    offspring.genome.id = next_genome_id()
                else:
                    parent1 = np.random.choice(survivors)
                    parent2 = np.random.choice(survivors)
                    
                    if parent1.fitness >= parent2.fitness:
                        dominant, recessive = parent1, parent2
                    else:
                        dominant, recessive = parent2, parent1
                    
                    offspring = genome_crossover(dominant, recessive)
                
                self.mutate(offspring.genome)
                new_generation.append(offspring)
        
        return new_generation

    def mutate(self, genome):
        if np.random.rand() < self.config.add_connection_rate:
            mutate_add_connection(genome)
        
        if np.random.rand() < self.config.add_node_rate:
            mutate_add_neuron(genome)
        
        if np.random.rand() < self.config.remove_connection_rate:
            mutate_remove_connection(genome)
        
        if np.random.rand() < self.config.remove_node_rate:
            mutate_remove_neuron(genome)
            
        if np.random.rand() < self.config.toggle_connection_rate:
            mutate_toggle_connection(genome)
        
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
        assert len(inputs) == len(self.input_ids)
        
        neuron_outputs = {}
        
        for i, input_value in enumerate(inputs):
            neuron_outputs[self.input_ids[i]] = input_value
        
        processed = set(self.input_ids)
        
        # Add maximum iteration limit to prevent infinite loops
        max_iterations = len(self.neurons) * 2  # Conservative limit
        iteration_count = 0
        
        while len(processed) < len(self.neurons) + len(self.input_ids):
            made_progress = False
            iteration_count += 1
            
            # Break if we've exceeded max iterations
            if iteration_count > max_iterations:
                break
            
            for neuron in self.neurons:
                if neuron.node_id in processed:
                    continue
                
                inputs_ready = all(inp.node_id in neuron_outputs for inp in neuron.inputs)
                
                if inputs_ready:
                    value = sum(neuron_outputs[inp.node_id] * inp.weight for inp in neuron.inputs)
                    value += neuron.bias

                    # Clamp extreme values to prevent overflow
                    value = np.clip(value, -100, 100)

                    if neuron.activation == "tanh":
                        neuron_outputs[neuron.node_id] = np.tanh(value)
                    elif neuron.activation == "sigmoid":
                        neuron_outputs[neuron.node_id] = 1 / (1 + np.exp(-value))
                    elif neuron.activation == "relu":
                        neuron_outputs[neuron.node_id] = max(0, value)
                    else:
                        neuron_outputs[neuron.node_id] = np.tanh(value)

                    processed.add(neuron.node_id)
                    made_progress = True
            
            if not made_progress:
                break

        outputs = []
        for output_id in self.output_ids:
            if output_id in neuron_outputs:
                outputs.append(neuron_outputs[output_id])
            else:
                outputs.append(0.0)
        
        return outputs
        
    @staticmethod
    def create_from_genome(genome):
        input_ids = [node.node_id for node in genome.get_input_nodes()]
        output_ids = [node.node_id for node in genome.get_output_nodes()]
        
        neurons = []
        for node in genome.nodes.values():
            if node.type != "input":
                inputs = []
                for conn in genome.connections.values():
                    if conn.id.to_node == node.node_id and conn.enabled:
                        inputs.append(NeuronInput(conn.id.from_node, conn.weight))
                
                if inputs:
                    neurons.append(Neuron(node, inputs))
        
        return FeedForwardNetwork(input_ids, output_ids, neurons)

def xor_fitness(genome):
    network = FeedForwardNetwork.create_from_genome(genome)
    
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_outputs = [0, 1, 1, 0]
    
    total_error = 0.0
    for inputs, expected in zip(xor_inputs, xor_outputs):
        output = network.activate(inputs)
        error = (output[0] - expected) ** 2
        total_error += error
    
    return 4.0 - total_error

class NEAT:
    def __init__(self, config, input_size, output_size):
        self.config = config
        self.population = Population(config, input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.output_size = output_size
    def run(self, fitness_function, num_generations):
        return self.population.run(fitness_function, num_generations)
    
def main():
    CONFIG = Config()
    num_generations = 1000

    neat = NEAT(CONFIG, 2, 1)
    winner = neat.run(xor_fitness, num_generations)

    print(f"\nBest genome (ID: {winner.genome.id}) with fitness: {winner.fitness:.4f}")
    print(f"Nodes: {len(winner.genome.nodes)}")
    print(f"Connections: {len(winner.genome.connections)}")
    
    network = FeedForwardNetwork.create_from_genome(winner.genome)
    print("\nTesting winner on XOR:")
    for inputs in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        output = network.activate(inputs)
        print(f"Input: {inputs}, Output: {output[0]:.4f}")
    
    return winner


class PopulationWithDynamicSpeciation(Population):
    def speciate(self):
        # Adjust threshold based on current species count
        target_species = max(5, self.config.population_size // 20)  # Target ~5-8 species
        
        if len(self.species) < target_species:
            self.config.compatibility_threshold *= 0.95  # Lower threshold
        elif len(self.species) > target_species * 1.5:
            self.config.compatibility_threshold *= 1.05  # Raise threshold
            
        # Ensure reasonable bounds
        self.config.compatibility_threshold = np.clip(
            self.config.compatibility_threshold, 0.5, 5.0
        )
        
        # Call original speciate method
        super().speciate()
        
        
if __name__ == "__main__":
    main()