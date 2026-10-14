import numpy as np
import torch
from pathlib import Path
from macro_place.benchmark import Benchmark

def force_directed(benchmark, adj, pos_map, 
                spring_constant=1,
                repulsion_constant=1, 
                step_size=0.01,
                num_iterations=20,
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
        
    for i, idx in enumerate(hard_indices):
        name = benchmark.macro_names[i]
        placement[idx, 0] = float(positions[i, 0])
        placement[idx, 1] = float(positions[i, 1])
        
    moved = sum(1 for i, idx in enumerate(hard_indices)
        if abs(placement[idx, 0].item() - positions[i, 0]) > 1e-6)
    print("macros whose FD result was written back:", len(hard_indices) - moved)
    placement = resolve_overlaps(benchmark, placement, hard_indices)
    
    
    return placement
    
    
def resolve_overlaps(benchmark, placement, hard_indices, max_passes=25):
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
    
# def force_directed_vec(benchmark, adj, pos_map, initial_placement=None,
#                        spring_constant=0.95, repulsion_constant=1.8,
#                        step_size=0.01, num_iterations=300,
#                        live_hard_neighbors=False, max_step_frac=0.02, device="cpu"):
#     base = initial_placement if initial_placement is not None else benchmark.macro_positions
#     placement = base.clone()
#     names = benchmark.macro_names
#     dt = torch.float64

#     hard_mask = benchmark.get_hard_macro_mask()
#     hard_indices = torch.where(hard_mask)[0].tolist()
#     n = len(hard_indices)
#     name_to_i = {names[idx]: i for i, idx in enumerate(hard_indices)}

#     pos = placement[hard_indices].to(dt).to(device)                  # [n, 2]
#     half = (benchmark.macro_sizes[hard_indices].to(dt) / 2).to(device)
#     canvas = torch.tensor([benchmark.canvas_width, benchmark.canvas_height],
#                           dtype=dt, device=device)
#     lo, hi = half, canvas - half
#     movable = benchmark.get_movable_mask()[hard_indices].to(device)  # fixed macros don't move

#     # ---- precompute attraction terms ----
#     A = torch.zeros(n, 2, dtype=dt)
#     Wsum = torch.zeros(n, dtype=dt)
#     rows, cols, vals = [], [], []
#     for i, idx in enumerate(hard_indices):
#         for nb, w in adj[names[idx]].items():
#             if live_hard_neighbors and nb in name_to_i:
#                 rows.append(i); cols.append(name_to_i[nb]); vals.append(w)
#             else:
#                 m = pos_map[nb]
#                 A[i, 0] += w * m.x
#                 A[i, 1] += w * m.y
#             Wsum[i] += w
#     A, Wsum = A.to(device), Wsum.to(device)
#     Wh = (torch.sparse_coo_tensor([rows, cols], vals, (n, n), dtype=dt, device=device)
#           if rows else None)

#     eye = torch.eye(n, dtype=torch.bool, device=device)

#     for _ in range(num_iterations):
#         # attraction
#         pull = A if Wh is None else A + torch.sparse.mm(Wh, pos)
#         f = spring_constant * (pull - Wsum[:, None] * pos)

#         # repulsion
#         diff = pos[:, None, :] - pos[None, :, :]          # [n, n, 2]
#         d2 = (diff ** 2).sum(-1)                          # [n, n]
#         valid = (~eye) & (d2 > 1e-12)
#         inv_d3 = torch.where(valid, d2.clamp_min(1e-12) ** -1.5, torch.zeros_like(d2))
#         f = f + repulsion_constant * (diff * inv_d3[..., None]).sum(dim=1)

#         # forces
#         pull = A if Wh is None else A + torch.sparse.mm(Wh, pos)
#         diff = pos[:, None, :] - pos[None, :, :]
#         d2 = (diff ** 2).sum(-1)
#         valid = (~eye) & (d2 > 1e-12)
#         inv_d3 = torch.where(valid, d2.clamp_min(1e-12) ** -1.5, torch.zeros_like(d2))
#         rep = repulsion_constant * (diff * inv_d3[..., None]).sum(dim=1)

#         # implicit attraction step (stable)
#         denom = (1 + step_size * spring_constant * Wsum)[:, None]
#         new = (pos + step_size * (spring_constant * pull + rep)) / denom

#         # cap per-iteration displacement
#         delta = new - pos
#         max_step = max_step_frac * canvas.min()
#         norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
#         delta = delta * torch.clamp(max_step / norm, max=1.0)
#         new = pos + delta

#         # freeze fixed macros, clamp to canvas
#         pos = torch.where(movable[:, None], new, pos)
#         pos = torch.max(torch.min(pos, hi), lo)

#     placement[hard_indices] = pos.to(placement.dtype).cpu()
#     return placement

def build_edges(plc, pos_map):
    names = list(pos_map.keys())
    name_to_k = {n: k for k, n in enumerate(names)}
    xy = np.array([[pos_map[n].x, pos_map[n].y] for n in names], dtype=np.float64)
    ea, eb, ew = [], [], []
    for m in plc.modules_w_pins:
        if m.get_type() != "MACRO":
            continue
        a = name_to_k[m.get_name()]
        for nb, w in m.get_connection().items():
            if nb in name_to_k:
                ea.append(a); eb.append(name_to_k[nb]); ew.append(w)
    return name_to_k, xy, np.array(ea), np.array(eb), np.array(ew, dtype=np.float64)


def rudy_map(xy, ea, eb, ew, W, H, grid_n):
    cw, ch = W / grid_n, H / grid_n
    C = np.zeros((grid_n, grid_n))
    x1, y1, x2, y2 = xy[ea, 0], xy[ea, 1], xy[eb, 0], xy[eb, 1]
    lx, hx = np.minimum(x1, x2), np.maximum(x1, x2)
    ly, hy = np.minimum(y1, y2), np.maximum(y1, y2)
    wb = np.maximum(hx - lx, cw)   # avoid zero-size boxes
    hb = np.maximum(hy - ly, ch)
    val = ew * (wb + hb) / (wb * hb)
    c0 = np.clip((lx / cw).astype(int), 0, grid_n - 1)
    c1 = np.clip((hx / cw).astype(int), 0, grid_n - 1)
    r0 = np.clip((ly / ch).astype(int), 0, grid_n - 1)
    r1 = np.clip((hy / ch).astype(int), 0, grid_n - 1)
    for k in range(len(val)):
        C[r0[k]:r1[k] + 1, c0[k]:c1[k] + 1] += val[k]

    # 3x3 box blur for a smoother gradient
    P = np.pad(C, 1, mode="edge")
    C = sum(P[1 + dr:1 + dr + grid_n, 1 + dc:1 + dc + grid_n]
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9.0

    C = C / max(C.mean(), 1e-12)
    gy, gx = np.gradient(C, ch, cw)    # dC/dy, dC/dx per micron
    return gx, gy

def rudy_nudge(benchmark, plc, pos_map, placement,
               alpha=0.5, iters=3, grid_n=32, move_soft=False):
    W, H = benchmark.canvas_width, benchmark.canvas_height
    cw, ch = W / grid_n, H / grid_n
    names = benchmark.macro_names
    name_to_k, all_xy, ea, eb, ew = build_edges(plc, pos_map)

    mask = benchmark.get_hard_macro_mask() & benchmark.get_movable_mask()
    if move_soft:
        mask = mask | benchmark.get_soft_macro_mask()
    idxs = torch.where(mask)[0].tolist()
    ks = np.array([name_to_k[names[i]] for i in idxs])

    pos = placement[idxs].numpy().astype(np.float64)
    half = benchmark.macro_sizes[idxs].numpy().astype(np.float64) / 2
    hi = np.array([W, H]) - half

    for _ in range(iters):
        all_xy[ks] = pos
        gx, gy = rudy_map(all_xy, ea, eb, ew, W, H, grid_n)
        cols = np.clip((pos[:, 0] / cw).astype(int), 0, grid_n - 1)
        rows = np.clip((pos[:, 1] / ch).astype(int), 0, grid_n - 1)
        g = np.stack([gx[rows, cols], gy[rows, cols]], axis=1)
        gmax = np.abs(g).max()
        if gmax < 1e-12:
            break
        pos -= alpha * np.array([cw, ch]) * g / gmax
        pos = np.clip(pos, half, hi)

    placement[idxs] = torch.tensor(pos, dtype=placement.dtype)
    return placement

def force_directed_cong(benchmark, plc, adj, pos_map,
                        spring_constant=1, repulsion_constant=1.2,
                        step_size=0.01, num_iterations=1,
                        k_cong=0.1, grid_n=32, rebuild_every=10,
                        warmup_frac=0.3, debug=True):
    placement = benchmark.macro_positions.clone()
    W, H = benchmark.canvas_width, benchmark.canvas_height
    cw, ch = W / grid_n, H / grid_n

    hard_mask = benchmark.get_hard_macro_mask()
    hard_indices = torch.where(hard_mask)[0].tolist()
    positions = placement[hard_indices].numpy()

    name_to_k, all_xy, ea, eb, ew = build_edges(plc, pos_map)
    hard_k = np.array([name_to_k[benchmark.macro_names[idx]] for idx in hard_indices])

    warmup = int(warmup_frac * num_iterations)
    gx = gy = None
    printed = False

    for it in range(num_iterations):
        if k_cong > 0 and it >= warmup and (it - warmup) % rebuild_every == 0:
            all_xy[hard_k] = positions
            gx, gy = rudy_map(all_xy, ea, eb, ew, W, H, grid_n)

        att_mags, cong_mags = [], []

        for i, idx in enumerate(hard_indices):
            force = np.zeros(2)
            name = benchmark.macro_names[idx]

            # attraction (same as original)
            f_att = np.zeros(2)
            for neighbor_name, weight in adj[name].items():
                nb = pos_map[neighbor_name]
                f_att += spring_constant * weight * (np.array([nb.x, nb.y]) - positions[i])
            force += f_att

            # repulsion (same as original)
            for j in range(len(hard_indices)):
                if j == i:
                    continue
                diff = positions[i] - positions[j]
                distance = np.linalg.norm(diff)
                if distance < 1e-6:
                    continue
                force += repulsion_constant / distance ** 2 * (diff / distance)

            # congestion
            if gx is not None:
                col = min(max(int(positions[i, 0] / cw), 0), grid_n - 1)
                row = min(max(int(positions[i, 1] / ch), 0), grid_n - 1)
                f_cong = -k_cong * np.array([gx[row, col], gy[row, col]])
                force += f_cong
                if debug and not printed:
                    att_mags.append(np.linalg.norm(f_att))
                    cong_mags.append(np.linalg.norm(f_cong))

            positions[i] += step_size * force

            hw = benchmark.macro_sizes[idx, 0].item() / 2
            hh = benchmark.macro_sizes[idx, 1].item() / 2
            positions[i, 0] = max(hw, min(positions[i, 0], W - hw))
            positions[i, 1] = max(hh, min(positions[i, 1], H - hh))

        if debug and not printed and cong_mags:
            print(f"[calib] median |F_att|={np.median(att_mags):.4f}  "
                  f"median |F_cong|={np.median(cong_mags):.4f}")
            printed = True

    for i, idx in enumerate(hard_indices):
        placement[idx, 0] = float(positions[i, 0])
        placement[idx, 1] = float(positions[i, 1])
    return placement