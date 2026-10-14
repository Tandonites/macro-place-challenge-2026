import numpy as np
import torch


def _edges(plc, pos_map):
    names = list(pos_map.keys())
    k_of = {n: k for k, n in enumerate(names)}
    xy = torch.tensor([[pos_map[n].x, pos_map[n].y] for n in names], dtype=torch.float64)
    ea, eb, ew = [], [], []
    for m in plc.modules_w_pins:
        if m.get_type() != "MACRO":
            continue
        a = k_of[m.get_name()]
        for nb, w in m.get_connection().items():
            if nb in k_of:
                ea.append(a); eb.append(k_of[nb]); ew.append(w)
    return (k_of, xy, torch.tensor(ea), torch.tensor(eb),
            torch.tensor(ew, dtype=torch.float64))


def _lse_wl(xy, ea, eb, ew, gamma):
    xa, xb = xy[ea], xy[eb]                                   # [E, 2]
    pos = torch.logaddexp(xa / gamma, xb / gamma)
    neg = torch.logaddexp(-xa / gamma, -xb / gamma)
    return (ew[:, None] * gamma * (pos + neg)).sum()


def _density_map(xy, half, G, W, H):
    bw, bh = W / G, H / G
    bx = torch.arange(G, dtype=xy.dtype) * bw
    by = torch.arange(G, dtype=xy.dtype) * bh
    x0, x1 = (xy[:, 0] - half[:, 0])[:, None], (xy[:, 0] + half[:, 0])[:, None]
    y0, y1 = (xy[:, 1] - half[:, 1])[:, None], (xy[:, 1] + half[:, 1])[:, None]
    ox = (torch.minimum(x1, bx + bw) - torch.maximum(x0, bx)).clamp(min=0)  # [n, G]
    oy = (torch.minimum(y1, by + bh) - torch.maximum(y0, by)).clamp(min=0)  # [n, G]
    return (oy.T @ ox) / (bw * bh)                                          # [G, G]


def analytical_place(benchmark, plc, pos_map,
                     iters=1500, grid_n=64, gamma_frac=0.01, lr_frac=0.002,
                     lam_scale=0.1, lam_mult=1.008, target_density=0.9, ov_scale=10.0,
                     move_soft=True, init="reference", seed=0, verbose=True):
    torch.manual_seed(seed)
    W, H = float(benchmark.canvas_width), float(benchmark.canvas_height)
    canvas = torch.tensor([W, H], dtype=torch.float64)
    k_of, all_xy, ea, eb, ew = _edges(plc, pos_map)
    names = benchmark.macro_names
    sizes = benchmark.macro_sizes.to(torch.float64)

    hard = benchmark.get_hard_macro_mask() & benchmark.get_movable_mask()
    soft = benchmark.get_soft_macro_mask()
    mov_mask = (hard | soft) if move_soft else hard
    mov_idx = torch.where(mov_mask)[0]
    fix_idx = torch.where(soft & ~mov_mask)[0]
    mov_k = torch.tensor([k_of[names[i]] for i in mov_idx.tolist()])
    half_m = sizes[mov_idx] / 2
    
    hard_sel = hard[mov_idx]                     # which movable objects are hard
    half_h = half_m[hard_sel]

    def hard_overlap(p):
        ph = p[hard_sel]
        dx = (half_h[:, None, 0] + half_h[None, :, 0]) - (ph[:, None, 0] - ph[None, :, 0]).abs()
        dy = (half_h[:, None, 1] + half_h[None, :, 1]) - (ph[:, None, 1] - ph[None, :, 1]).abs()
        return (dx.clamp(min=0) * dy.clamp(min=0)).triu(1).sum()

    # density of soft macros that stay fixed
    if len(fix_idx):
        fixed_map = _density_map(benchmark.macro_positions[fix_idx].double(),
                                 sizes[fix_idx] / 2, grid_n, W, H)
    else:
        fixed_map = torch.zeros(grid_n, grid_n, dtype=torch.float64)

    # initial positions
    if init == "center":
        p0 = canvas / 2 + torch.randn(len(mov_idx), 2, dtype=torch.float64) * 0.05 * canvas
    else:
        p0 = benchmark.macro_positions[mov_idx].double()
    p0 = torch.max(torch.min(p0, canvas - half_m), half_m)
    p = p0.clone().requires_grad_(True)

    gamma = gamma_frac * (W + H) / 2
    opt = torch.optim.Adam([p], lr=lr_frac * (W + H) / 2)

    def losses(p):
        xy = all_xy.index_put((mov_k,), p)
        wl = _lse_wl(xy, ea, eb, ew, gamma)
        dmap = fixed_map + _density_map(p, half_m, grid_n, W, H)
        dpen = ((dmap - target_density).clamp(min=0) ** 2).sum()
        return wl, dpen

    # calibrate lambda so density starts at lam_scale of wirelength's pull
    wl, dpen = losses(p)
    gw = torch.autograd.grad(wl, p, retain_graph=True)[0].norm()
    gd = torch.autograd.grad(dpen, p)[0].norm()
    lam = lam_scale * (gw / gd.clamp(min=1e-12)).item()

    for t in range(iters):
        opt.zero_grad()
        wl, dpen = losses(p)
        (wl + lam * dpen + lam * ov_scale * hard_overlap(p)).backward()
        opt.step()
        with torch.no_grad():
            p.copy_(torch.max(torch.min(p, canvas - half_m), half_m))
        lam *= lam_mult
        if verbose and t % 250 == 0:
            print(f"[ap] it {t:4d}  wl {wl.item():.1f}  overflow {dpen.item():.3f}  lam {lam:.3g}")

    placement = benchmark.macro_positions.clone()
    placement[mov_idx] = p.detach().to(placement.dtype)
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