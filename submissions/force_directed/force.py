import numpy as np
import torch
from pathlib import Path
from macro_place.benchmark import Benchmark

def force_directed(benchmark, adj, pos_map, 
                spring_constant=1.0,
                repulsion_constant=1.0, 
                step_size=0.1,
                num_iterations=100,
                seed=42):
    placement = benchmark.macro_positions.clone()
    rng = np.random.default_rng(seed)
    
    hard_mask = benchmark.get_hard_macro_mask()
    hard_indices = torch.where(hard_mask)[0].tolist()
    
    positions = placement[hard_indices].numpy()
    
    for iteration in range(num_iterations):
        for i, idx in enumerate(hard_indices):
            force = np.zeros(2)
            name = benchmark.macro_names[idx]
            neighbors = adj[name]
            for neighbor_name, weight in neighbors.items():
                neighbor = pos_map[neighbor_name]
                current_pos = positions[i]
                neighbor_pos = np.array([neighbor.x, neighbor.y])
                direction = neighbor_pos - current_pos
                force += spring_constant * weight * direction
            for j, jdx in enumerate(hard_indices):
                if j != i:
                    diff = positions[i] - positions[j]
                    distance = np.linalg.norm(diff)
                    if distance < 1e-6:
                        continue
                    unit_vector_away = diff / distance
                    force += repulsion_constant / distance**2 * unit_vector_away     
            
            positions[i] += step_size * force
            
            half_w = benchmark.macro_sizes[idx, 0].item() / 2
            half_h = benchmark.macro_sizes[idx, 1].item() / 2
            positions[i, 0] = max(half_w, min(positions[i, 0], benchmark.canvas_width - half_w))
            positions[i, 1] = max(half_h, min(positions[i, 1], benchmark.canvas_height - half_h))
        
    for i in hard_indices:
        name = benchmark.macro_names[i]
        placement[idx, 0] = float(positions[i, 0])
        placement[idx, 1] = float(positions[i, 1])
    return placement
    
    
