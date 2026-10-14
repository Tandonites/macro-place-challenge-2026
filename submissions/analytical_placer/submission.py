import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from analytical import analytical_place, resolve_overlaps, legalize

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

class AnalyticalPlacer:
    def place(self, benchmark):
        plc = _load_plc(benchmark.name)
        pos_map = {m.get_name(): m for m in plc.modules_w_pins}
        hard_indices = torch.where(benchmark.get_hard_macro_mask())[0].tolist()

        placement = analytical_place(benchmark, plc, pos_map)
        placement = resolve_overlaps(benchmark, placement, hard_indices)
        placement = legalize(benchmark, placement)
        return placement