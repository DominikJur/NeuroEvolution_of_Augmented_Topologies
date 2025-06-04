# NEAT Don't Touch the Spikes

An AI-driven implementation of the classic "Don't Touch the Spikes" game using NEAT (NeuroEvolution of Augmenting Topologies) neural networks. Watch as artificial agents learn to navigate through increasingly challenging spike patterns, evolving their strategies over generations.

## 🎮 About the Game

"Don't Touch the Spikes" is a simple yet challenging arcade game where players control a bouncing character that must avoid touching deadly spikes while bouncing between walls. The objective is to survive as long as possible, collecting coins and achieving high scores.

## 🧠 NEAT Integration

This project implements the NEAT algorithm to train AI agents that can play the game autonomously. NEAT is a powerful evolutionary algorithm that:

- **Evolves Neural Network Topologies**: Unlike traditional neural networks with fixed structures, NEAT evolves both the weights and the structure of the neural networks
- **Maintains Genetic Diversity**: Through speciation, NEAT protects innovative solutions and maintains population diversity
- **Gradually Complexifies**: Networks start simple and gradually add complexity only when beneficial
- **Historical Markings**: Tracks genetic innovations to enable meaningful crossover between different network topologies

## 🔬 Key Features

### Evolutionary Learning
- **Population-based Evolution**: Multiple agents learn simultaneously through generations
- **Fitness-based Selection**: Agents are evaluated based on survival time and score
- **Structural Mutations**: Networks can add/remove nodes and connections
- **Parameter Optimization**: Weights and biases evolve through mutation and crossover

### Game Integration
- **Real-time Training**: Watch agents learn and improve in real-time
- **Multiple Game Modes**: Single-player and multi-player training environments
- **Configurable Parameters**: Extensive configuration options for both game and NEAT settings
- **Performance Tracking**: Monitor fitness evolution and network complexity

### Neural Network Architecture
- **Input Processing**: Agents receive information about player position, velocity, gravity, and spike locations
- **Dynamic Topology**: Network structure evolves based on performance needs
- **Activation Functions**: Support for various activation functions (tanh, sigmoid, ReLU)
- **Feed-forward Processing**: Efficient neural network evaluation for real-time gameplay

## 🏗️ Project Structure

```
├── src/
│   ├── game.py              # Core game logic and mechanics
│   ├── multi_player_game.py # Multi-agent training environment
│   ├── neat.py              # NEAT algorithm implementation
│   ├── player.py            # Player character with physics
│   ├── spike.py             # Spike obstacle classes
│   └── coin.py              # Collectible coin system
├── graphics/                # Sprite assets and visual resources
├── audio/                   # Sound effects and audio files
├── data/                    # Persistent game data (scores, coins)
├── config.py                # Configuration settings for NEAT and game
└── main.py                  # Entry point for training and gameplay
```

## 🎯 Learning Objectives

This project demonstrates several important concepts in artificial intelligence and evolutionary computation:

1. **Neuroevolution**: How neural networks can be evolved rather than trained through backpropagation
2. **Genetic Algorithms**: Population-based optimization and selection strategies
3. **Emergent Behavior**: How complex behaviors can emerge from simple rules and interactions
4. **Real-time AI**: Integration of AI algorithms with interactive applications
5. **Performance Optimization**: Balancing computational efficiency with learning effectiveness

## 🔧 Technical Implementation

### NEAT Algorithm Components
- **Genome Representation**: Efficient encoding of neural network structure and parameters
- **Innovation Tracking**: Global innovation numbers for consistent genetic operations
- **Speciation**: Automatic grouping of similar genomes to protect diversity
- **Crossover Operations**: Structure-aware recombination of parent genomes
- **Mutation Operators**: Structural and parametric mutations for exploration

### Game Physics
- **Collision Detection**: Precise collision handling for spikes, walls, and coins
- **Gravity System**: Realistic physics simulation for player movement
- **Dynamic Difficulty**: Spike generation that adapts to player performance
- **State Management**: Comprehensive game state tracking for AI input

### Performance Features
- **Headless Mode**: Training without visual rendering for faster evolution
- **Batch Processing**: Efficient evaluation of multiple agents simultaneously
- **Configurable Parameters**: Easy tuning of learning rates and game mechanics
- **Progress Monitoring**: Real-time statistics and evolutionary progress tracking

## 📊 Training Insights

The AI agents learn through trial and error, developing strategies such as:
- **Timing Optimization**: Learning when to jump for maximum survival
- **Pattern Recognition**: Identifying safe zones and spike arrangements
- **Risk Assessment**: Balancing coin collection with survival priorities
- **Adaptive Strategies**: Adjusting behavior based on current game state

## 🔬 Research Applications

This implementation serves as an excellent platform for:
- **Algorithm Comparison**: Testing NEAT against other evolutionary approaches
- **Hyperparameter Studies**: Analyzing the impact of various NEAT parameters
- **Behavioral Analysis**: Understanding emergent AI strategies and decision-making
- **Educational Demonstrations**: Teaching concepts in AI and evolutionary computation

## 📈 Future Enhancements

Potential areas for expansion include:
- **HyperNEAT Integration**: Evolving network patterns for larger, more complex topologies
- **Multi-objective Optimization**: Balancing multiple fitness criteria simultaneously
- **Transfer Learning**: Applying learned behaviors to similar games or environments
- **Visualization Tools**: Enhanced monitoring and analysis of evolutionary progress
- **Competitive Evolution**: Agent vs. agent learning scenarios

## 🎓 Educational Value

This project provides hands-on experience with:
- Evolutionary algorithms and genetic programming
- Neural network topology optimization
- Real-time AI system integration
- Game development and physics simulation
- Performance analysis and optimization techniques

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Explore the fascinating world of neuroevolution and watch as simple algorithms learn to master complex challenges through the power of evolution and neural networks.*
