import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from graph import build_weight_graph
from force import force_directed, legalize, force_directed_cong, rudy_nudge, resolve_overlaps


def _load_plc(name):
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


class ForceDirectedPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark.name)
        pos_map = {m.get_name(): m for m in plc.modules_w_pins}
        adj = build_weight_graph(benchmark, plc)
        
        hard_indices = torch.where(benchmark.get_hard_macro_mask())[0].tolist()
        
        placement = benchmark.macro_positions.clone()
        placement = rudy_nudge(benchmark, plc, pos_map, placement, alpha=0, iters=5)
        placement = resolve_overlaps(benchmark, placement, hard_indices)
        placement = legalize(benchmark, placement)
        
        # placement = force_directed(benchmark, adj, pos_map)
        # placement = force_directed_vec(benchmark, adj, pos_map)
        # placement = force_directed_cong(benchmark, plc, adj, pos_map, k_cong=0)
        # placement = legalize(benchmark, placement)
        return placement