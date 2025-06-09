import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

def visualize_genome_topology(genome, title="Neural Network Topology", figsize=(16, 10), save_path=None, input_labels=None):
    
    if input_labels is None:
        # UPDATED: New input labels for 8 inputs
        input_labels = [
            'distance_to_wall',      # 0 - Distance to next wall
            'player_y',              # 1 - Normalized player Y position  
            'gravity',               # 2 - NEW: Gravity state for timing
            'spikes_way_above',      # 3 - NEW: Way above player
            'spikes_above',          # 4 - NEW: Above player
            'spikes_at_level',       # 5 - At player level
            'spikes_below',          # 6 - NEW: Below player
            'spikes_way_below',      # 7 - NEW: Way below player
        ]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    input_nodes = sorted([n for n in genome.nodes.values() if n.type == "input"], key=lambda x: x.node_id)
    output_nodes = sorted([n for n in genome.nodes.values() if n.type == "output"], key=lambda x: x.node_id)
    hidden_nodes = sorted([n for n in genome.nodes.values() if n.type == "hidden"], key=lambda x: x.node_id)
    
    max_nodes = max(len(input_nodes), len(output_nodes), len(hidden_nodes) if hidden_nodes else 0)
    height = max_nodes * 1.5
    width = 10
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    node_positions = {}
    
    if input_nodes:
        spacing = height / (len(input_nodes) + 1)
        for i, node in enumerate(input_nodes):
            y = spacing * (i + 1)
            node_positions[node.node_id] = (1, y)
    
    if output_nodes:
        spacing = height / (len(output_nodes) + 1)
        for i, node in enumerate(output_nodes):
            y = spacing * (i + 1)
            node_positions[node.node_id] = (width - 1, y)
    
    if hidden_nodes:
        cols = max(1, len(hidden_nodes) // 5)
        for i, node in enumerate(hidden_nodes):
            col = i // 5
            row = i % 5
            x = 3 + col * 2
            y = height / 6 * (row + 1)
            node_positions[node.node_id] = (x, y)
    
    for conn in genome.connections.values():
        if conn.enabled and conn.id.from_node in node_positions and conn.id.to_node in node_positions:
            from_pos = node_positions[conn.id.from_node]
            to_pos = node_positions[conn.id.to_node]
            
            thickness = min(3, max(0.5, abs(conn.weight)))
            color = 'red' if conn.weight < 0 else 'blue'
            alpha = min(0.8, abs(conn.weight) / 3 + 0.3)
            
            ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]], 
                   color=color, linewidth=thickness, alpha=alpha)
            
            if abs(conn.weight) > 2:
                mid_x = (from_pos[0] + to_pos[0]) / 2
                mid_y = (from_pos[1] + to_pos[1]) / 2
                ax.text(mid_x, mid_y, f'{conn.weight:.1f}', 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    for node in genome.nodes.values():
        if node.node_id in node_positions:
            pos = node_positions[node.node_id]
            
            if node.type == "input":
                color = 'lightgreen'
                size = 300
            elif node.type == "output":
                color = 'lightcoral'
                size = 300
            else:
                color = 'lightblue'
                size = 200
            
            ax.scatter(pos[0], pos[1], s=size, c=color, edgecolors='black', linewidth=2, zorder=3)
            ax.text(pos[0], pos[1], str(node.node_id), ha='center', va='center', fontweight='bold', zorder=4)
            
            if node.type != "input":
                ax.text(pos[0], pos[1] - 0.3, f'b:{node.bias:.1f}', 
                       ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='gray', alpha=0.7))
    
    for i, node in enumerate(input_nodes):
        pos = node_positions[node.node_id]
        label_idx = -(node.node_id + 1)
        if 0 <= label_idx < len(input_labels):
            label = input_labels[label_idx]
        else:
            label = f"input_{label_idx}"
        
        ax.text(pos[0] - 0.8, pos[1], label, ha='right', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))
    
    for node in output_nodes:
        pos = node_positions[node.node_id]
        ax.text(pos[0] + 0.8, pos[1], 'jump_action', ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.8))
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Input (8)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=8, label='Hidden'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Output (1)'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Positive Weight'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Negative Weight')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    info_text = f"""Genome {genome.id}
Fitness: {getattr(genome, 'fitness', 'N/A')}
Nodes: {len(genome.nodes)} ({len(input_nodes)}+{len(hidden_nodes)}+{len(output_nodes)})
Connections: {len([c for c in genome.connections.values() if c.enabled])}/{len(genome.connections)}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def print_genome_details(genome):
    print(f"\n{'='*50}")
    print(f"GENOME {genome.id} DETAILS")
    print(f"{'='*50}")
    print(f"Fitness: {getattr(genome, 'fitness', 'N/A')}")
    print(f"Total Nodes: {len(genome.nodes)}")
    print(f"Total Connections: {len(genome.connections)}")
    print(f"Enabled Connections: {len([c for c in genome.connections.values() if c.enabled])}")
    
    # UPDATED: New input mapping
    input_mapping = {
        -1: "distance_to_wall",
        -2: "player_y (norm)",
        -3: "gravity",
        -4: "spikes_way_above",
        -5: "spikes_above",
        -6: "spikes_at_level",
        -7: "spikes_below",
        -8: "spikes_way_below"
    }
    
    print(f"\nINPUT NODES:")
    for node in sorted([n for n in genome.nodes.values() if n.type == "input"], key=lambda x: x.node_id):
        description = input_mapping.get(node.node_id, f"input_{-(node.node_id+1)}")
        print(f"  {node.node_id}: {description}")
    
    print(f"\nOUTPUT NODES:")
    for node in sorted([n for n in genome.nodes.values() if n.type == "output"], key=lambda x: x.node_id):
        print(f"  {node.node_id}: jump_action (bias: {node.bias:.3f})")
    
    if any(n.type == "hidden" for n in genome.nodes.values()):
        print(f"\nHIDDEN NODES:")
        for node in sorted([n for n in genome.nodes.values() if n.type == "hidden"], key=lambda x: x.node_id):
            print(f"  {node.node_id}: bias={node.bias:.3f}")
    
    print(f"\nSTRONG CONNECTIONS (|weight| > 1.0):")
    strong_connections = [c for c in genome.connections.values() if c.enabled and abs(c.weight) > 1.0]
    for conn in sorted(strong_connections, key=lambda x: abs(x.weight), reverse=True):
        from_desc = input_mapping.get(conn.id.from_node, f"node_{conn.id.from_node}")
        to_desc = "jump_action" if conn.id.to_node == 0 else f"node_{conn.id.to_node}"
        print(f"  {from_desc} → {to_desc}: {conn.weight:.2f}")

def analyze_genome_behavior(genome):
    print(f"\nBEHAVIOR ANALYSIS:")
    
    direct_connections = [c for c in genome.connections.values() 
                         if c.enabled and c.id.from_node < 0 and c.id.to_node == 0]
    
    if direct_connections:
        print("Direct input influences on jumping:")
        
        player_y_weight = sum(c.weight for c in direct_connections if c.id.from_node == -2)
        gravity_weight = sum(c.weight for c in direct_connections if c.id.from_node == -3)
        spike_weight = sum(c.weight for c in direct_connections if c.id.from_node in [-4, -5, -6, -7, -8])
        
        if player_y_weight != 0:
            print("  ✅ Responds to player Y position")
        if gravity_weight != 0:
            print("  ✅ Responds to gravity (timing)")
        if spike_weight != 0:
            spike_behavior = "avoids" if spike_weight < 0 else "attracts to"
            print(f"  ⚠️  {spike_behavior} spikes")
    else:
        print("No direct connections - uses hidden layer processing")
    
    complexity = len([n for n in genome.nodes.values() if n.type == "hidden"])
    if complexity == 0:
        print("Simple linear model - fast decisions")
    elif complexity < 3:
        print("Moderate complexity")
    else:
        print("High complexity - sophisticated behavior")

def visualize_best_genome(filename="best_genome.pkl"):
    import pickle
    import os
    
    genome_path = os.path.join("models", filename)
    
    if not os.path.exists(genome_path):
        print(f"File not found: {genome_path}")
        return
    
    try:
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
        
        print_genome_details(genome)
        analyze_genome_behavior(genome)
        visualize_genome_topology(genome, f"Best Genome {genome.id}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    visualize_best_genome()