import numpy as np
import torch
from pathlib import Path
from macro_place.benchmark import Benchmark

def force_directed(benchmark, adj, pos_map, 
                spring_constant=0.95,
                repulsion_constant=1.8, 
                step_size=0.01,
                num_iterations=200,
                seed=67):
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
    
    placement = resolve_overlaps(benchmark, placement, hard_indices)
    
    return placement
    
    
def resolve_overlaps(benchmark, placement, hard_indices, max_passes=50):
    for pass_num in range(max_passes):
        overlap_count = 0
        
        for i, idx_i in enumerate(hard_indices):
            x_i = placement[idx_i, 0].item()
            y_i = placement[idx_i, 1].item()
            half_w_i = benchmark.macro_sizes[idx_i, 0].item() / 2
            half_h_i = benchmark.macro_sizes[idx_i, 1].item() / 2
            
            for j, idx_j in enumerate(hard_indices):
                if j <= i:
                    continue
                    
                x_j = placement[idx_j, 0].item()
                y_j = placement[idx_j, 1].item()
                half_w_j = benchmark.macro_sizes[idx_j, 0].item() / 2
                half_h_j = benchmark.macro_sizes[idx_j, 1].item() / 2
                
                # compute overlap in x and y
                overlap_x = (half_w_i + half_w_j) - abs(x_i - x_j)
                overlap_y = (half_h_i + half_h_j) - abs(y_i - y_j)
                
                if overlap_x > 0 and overlap_y > 0:
                    overlap_count += 1
                    
                    # push apart along axis of minimum overlap
                    if overlap_x < overlap_y:
                        push = overlap_x / 2 + 0.001
                        if x_i < x_j:
                            placement[idx_i, 0] -= push
                            placement[idx_j, 0] += push
                        else:
                            placement[idx_i, 0] += push
                            placement[idx_j, 0] -= push
                    else:
                        push = overlap_y / 2 + 0.001
                        if y_i < y_j:
                            placement[idx_i, 1] -= push
                            placement[idx_j, 1] += push
                        else:
                            placement[idx_i, 1] += push
                            placement[idx_j, 1] -= push
                    
                    # clamp to canvas bounds
                    for idx in [idx_i, idx_j]:
                        hw = benchmark.macro_sizes[idx, 0].item() / 2
                        hh = benchmark.macro_sizes[idx, 1].item() / 2
                        placement[idx, 0] = max(hw, min(placement[idx, 0].item(), benchmark.canvas_width - hw))
                        placement[idx, 1] = max(hh, min(placement[idx, 1].item(), benchmark.canvas_height - hh))
        
        print(f"Pass {pass_num + 1}: {overlap_count} overlaps")
        if overlap_count == 0:
            break
    
    return placement
    
def place_isolated(benchmark, placement, hard_indices, adj):
    # build a grid to count macro density
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))
    
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    
    # count how many macros are in each grid cell
    for i, idx in enumerate(hard_indices):
        x = placement[idx, 0].item()
        y = placement[idx, 1].item()
        col = min(int(x / canvas_w * grid_size), grid_size - 1)
        row = min(int(y / canvas_h * grid_size), grid_size - 1)
        grid[row][col] += 1
    
    # find and place isolated macros
    for i, idx in enumerate(hard_indices):
        name = benchmark.macro_names[idx]
        if len(adj[name]) == 0:
            # find emptiest cell
            row, col = np.unravel_index(np.argmin(grid), grid.shape)
            
            # place macro at center of that cell
            half_w = benchmark.macro_sizes[idx, 0].item() / 2
            half_h = benchmark.macro_sizes[idx, 1].item() / 2
            
            x = (col + 0.5) / grid_size * canvas_w
            y = (row + 0.5) / grid_size * canvas_h
            
            x = max(half_w, min(x, canvas_w - half_w))
            y = max(half_h, min(y, canvas_h - half_h))
            
            placement[idx, 0] = float(x)
            placement[idx, 1] = float(y)
            
            # mark that cell as more populated
            grid[row][col] += 1
    
    return placement

def legalize(benchmark, placement, step_frac=0.005, max_rings=400, gap=0.002):
    sizes = benchmark.macro_sizes
    W, H = benchmark.canvas_width, benchmark.canvas_height
    step = min(W, H) * step_frac

    hard = benchmark.get_hard_macro_mask()
    movable = benchmark.get_movable_mask()
    fixed_idx = torch.where(hard & ~movable)[0].tolist()
    move_idx = torch.where(hard & movable)[0].tolist()

    # already-placed macros: x, y, half_w, half_h
    px, py, phw, phh = [], [], [], []
    for idx in fixed_idx:
        px.append(placement[idx, 0].item()); py.append(placement[idx, 1].item())
        phw.append(sizes[idx, 0].item() / 2); phh.append(sizes[idx, 1].item() / 2)

    def is_free(cx, cy, hw, hh):
        if not px:
            return True
        ox = (hw + np.array(phw) + gap) - np.abs(cx - np.array(px))
        oy = (hh + np.array(phh) + gap) - np.abs(cy - np.array(py))
        return not np.any((ox > 0) & (oy > 0))

    order = sorted(move_idx, key=lambda i: -(sizes[i, 0] * sizes[i, 1]).item())

    for idx in order:
        hw = sizes[idx, 0].item() / 2
        hh = sizes[idx, 1].item() / 2
        tx = placement[idx, 0].item()
        ty = placement[idx, 1].item()
        found = None

        for r in range(max_rings):
            # grid points on the square ring at radius r, nearest first
            if r == 0:
                ring = [(0, 0)]
            else:
                ring = [(dx, dy) for dx in range(-r, r + 1) for dy in (-r, r)]
                ring += [(dx, dy) for dx in (-r, r) for dy in range(-r + 1, r)]
                ring.sort(key=lambda d: d[0] ** 2 + d[1] ** 2)

            for dx, dy in ring:
                cx = min(max(tx + dx * step, hw), W - hw)
                cy = min(max(ty + dy * step, hh), H - hh)
                if is_free(cx, cy, hw, hh):
                    found = (cx, cy)
                    break
            if found:
                break

        if found is None:
            raise RuntimeError(f"Could not legalize macro {benchmark.macro_names[idx]}")

        placement[idx, 0] = float(found[0])
        placement[idx, 1] = float(found[1])
        px.append(found[0]); py.append(found[1]); phw.append(hw); phh.append(hh)

    return placement
    