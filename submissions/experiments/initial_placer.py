# OUTLINE

# use spatial hashing to achieve O(N) collision checking
# place macros in descending order of number of connections
# take workload off of SA refinement

# target score: 1.4 -> ~1.2 w/SA
# reach score: 1.3 -> ~1.1 w/SA

import torch
from macro_place.loader import load_benchmark_from_dir

filename = "submissions/experiments/spacehash_params.txt"

with open(filename, 'w') as f:
    print("spacehash min tile sizes\n", file=f)

    for i in range(1, 19):
        if i == 5:
            continue # idfk why there's no ibm05

        benchmark, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/ibm{i:02}")

        hard_mask = benchmark.get_hard_macro_mask()
        hard_dimensions = benchmark.macro_sizes[hard_mask]
        print(f"ibm{i:02}: {hard_dimensions[:,0].max():.1f} x {hard_dimensions[:,1].max():.1f}", file=f)
