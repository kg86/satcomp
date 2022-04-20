# Computing NP-hard Repetitiveness Measure by SAT solvers

Computes the smallest
- a smallest smallest string attractor,
- a smallest straight-line program, and
- a bidirectional macro scheme with the fewest phrases,
by using the SAT solver pySAT.

## Motivation

Computing these compression measures is NP-hard.
Solving such hard problems with a SAT-solver seems therefore natural.

## Automatic Evaluation

The script `shell/example_run.sh` evaluates our solutions on files of the Canterbury and Calgary corpus.
On success, it creates a JSON file `shell/example_run/benchmark.json` storing various statistics of the conducted benchmark.
Depending on the used operating system, this script might fail.
For that reasons, we provide a Docker image.
The starting point for that is the script `shell/docker/gen.sh`, 
which builds a Docker image, and runs the code inside a Docker container.
On success, it puts a `shell/docker/plot.tar` file back onto the host machine.
This file contains LaTex code with tikz/pgf instructions, 
which can generate plots for the aforementioned datasets.

## Build instructions

1. Install `pipenv` and `python3` on your OS.
2. Run the following command at the root of this repository.

```bash
$ pipenv sync
```

### Notes

The code has been tested only on Linux.

## Usage

Our executables are written in Python, and can be accessed from the `src` directory.

You can run the programs via the `pipenv` command as follows.

```bash
$ pipenv run python src/attractor_solver.py -h
usage: attractor_solver.py [-h] [--file FILE] [--str STR] [--output OUTPUT] [--contains CONTAINS [CONTAINS ...]] [--size SIZE] [--algo ALGO] [--log_level LOG_LEVEL]

Compute Minimum String Attractors.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           input file
  --str STR             input string
  --output OUTPUT       output file
  --contains CONTAINS [CONTAINS ...]
                        list of text positions that must be included in the string attractor, starting with index 1
  --size SIZE           exact size or upper bound of attractor size to search
  --algo ALGO           [min: find a minimum string attractor, exact/atmost: find a string attractor whose size is exact/atmost SIZE]
  --log_level LOG_LEVEL
                        log level, DEBUG/INFO/CRITICAL
```

## Running Examples


### Computing minimum string attractors with PySAT

The following program compute a minimum string attractor of size `factor_size = 497` for the dataset `grammar.lsp` having a length of 3721.

```bash
$ pipenv run python src/attractor_solver.py --file data/cantrbry/grammar.lsp --algo min
{"date": "2022-04-19 20:42:38.553750", "status": "", "algo": "attractor-sat", "file_name": "grammar.lsp", "file_len": 3721, "time_prep": 0.08829855918884277, "time_total": 0.38588380813598633, "sol_nvars": 3721, "sol_nhard": 1669, "sol_nsoft": 3721, "sol_navgclause": 17.860395446375076, "sol_ntotalvars": 29809, "sol_nmaxclause": 802, "factor_size": 497, "factors": [2, 5, 8, 10, 12, 14, 20, 22, 25, 28, 31, 34, 38, 44, 50, 51, 54, 57, 59, 62, 65, 69, 72, 77, 80, 85, 95, 99, 108, 114, 118, 121, 125, 133, 139, 143, 155, 161, 170, 181, 183, 186, 194, 201, 226, 239, 255, 261, 268, 276, 289, 294, 298, 307, 310, 313, 326, 341, 346, 356, 358, 378, 411, 429, 440, 449, 454, 472, 482, 489, 510, 523, 535, 546, 591, 596, 628, 650, 675, 677, 688, 707, 719, 725, 741, 759, 766, 785, 796, 805, 829, 845, 854, 872, 913, 917, 930, 941, 959, 981, 1005, 1011, 1018, 1029, 1031, 1051, 1053, 1064, 1082, 1103, 1108, 1118, 1129, 1149, 1167, 1176, 1186, 1189, 1216, 1224, 1234, 1253, 1272, 1275, 1280, 1286, 1292, 1294, 1300, 1305, 1316, 1321, 1327, 1335, 1346, 1354, 1369, 1377, 1400, 1413, 1430, 1432, 1435, 1482, 1498, 1512, 1552, 1573, 1584, 1585, 1587, 1591, 1613, 1629, 1631, 1644, 1670, 1717, 1732, 1738, 1742, 1744, 1746, 1749, 1752, 1754, 1762, 1770, 1776, 1782, 1786, 1788, 1792, 1795, 1798, 1804, 1806, 1809, 1811, 1815, 1819, 1821, 1825, 1831, 1841, 1851, 1857, 1862, 1868, 1874, 1880, 1885, 1891, 1898, 1901, 1904, 1913, 1915, 1918, 1925, 1928, 1933, 1942, 1952, 1956, 1962, 1964, 1966, 1972, 1978, 1983, 2000, 2005, 2013, 2017, 2022, 2028, 2031, 2040, 2044, 2056, 2062, 2074, 2076, 2085, 2089, 2101, 2106, 2109, 2112, 2119, 2123, 2124, 2133, 2143, 2147, 2150, 2153, 2157, 2159, 2165, 2168, 2175, 2177, 2178, 2183, 2190, 2197, 2206, 2212, 2214, 2216, 2220, 2224, 2226, 2228, 2230, 2239, 2243, 2248, 2252, 2256, 2261, 2266, 2277, 2281, 2290, 2295, 2299, 2303, 2309, 2312, 2316, 2321, 2335, 2340, 2344, 2360, 2362, 2366, 2379, 2383, 2386, 2401, 2404, 2418, 2428, 2448, 2457, 2470, 2472, 2491, 2496, 2500, 2516, 2524, 2536, 2540, 2558, 2566, 2569, 2572, 2584, 2587, 2601, 2613, 2622, 2627, 2631, 2639, 2649, 2657, 2664, 2680, 2690, 2693, 2696, 2705, 2718, 2721, 2741, 2745, 2749, 2762, 2766, 2777, 2793, 2797, 2808, 2815, 2825, 2836, 2840, 2844, 2859, 2868, 2871, 2894, 2900, 2902, 2916, 2918, 2921, 2927, 2943, 2964, 2968, 2974, 2990, 2993, 2998, 3004, 3017, 3023, 3026, 3037, 3051, 3056, 3065, 3068, 3070, 3074, 3085, 3092, 3099, 3103, 3115, 3119, 3122, 3131, 3139, 3142, 3147, 3150, 3152, 3159, 3164, 3165, 3177, 3184, 3197, 3202, 3205, 3209, 3216, 3220, 3222, 3230, 3235, 3241, 3249, 3252, 3257, 3259, 3264, 3268, 3271, 3278, 3281, 3288, 3295, 3298, 3301, 3305, 3314, 3319, 3324, 3328, 3335, 3336, 3348, 3360, 3366, 3369, 3374, 3377, 3385, 3391, 3394, 3398, 3405, 3414, 3420, 3422, 3428, 3431, 3440, 3443, 3449, 3454, 3458, 3465, 3467, 3471, 3474, 3477, 3480, 3483, 3486, 3490, 3493, 3500, 3504, 3515, 3519, 3521, 3526, 3530, 3534, 3544, 3550, 3555, 3558, 3560, 3564, 3570, 3573, 3579, 3581, 3585, 3590, 3592, 3594, 3598, 3603, 3606, 3609, 3612, 3624, 3626, 3627, 3631, 3634, 3636, 3639, 3642, 3644, 3645, 3647, 3649, 3651, 3653, 3655, 3662, 3668, 3671, 3676, 3677, 3681, 3685, 3689, 3692, 3694, 3697, 3701, 3705, 3712]}
```

### Computing minimum bidirectional macroschemes with PySAT

The following program will compute minimum bidirectional macroschemes of size `factor_size = 43` for `cp.html-50` of length 50.

```bash
pipenv run python src/bidirectional_solver.py --file data/cantrbry_pref/cp.html-50 
{"date": "2022-04-19 20:51:17.510793", "status": "", "algo": "bidirectional-sat", "file_name": "cp.html-50", "file_len": 50, "time_prep": 0.010410785675048828, "time_total": 0.014687776565551758, "sol_nvars": 1140, "sol_nhard": 3517, "sol_nsoft": 50, "sol_navgclause": 2.3810065396644866, "sol_ntotalvars": 8374, "sol_nmaxclause": 761, "factor_size": 43, "factors": [[-1, 60], [-1, 104], [-1, 101], [-1, 97], [-1, 100], [41, 3], [-1, 116], [-1, 105], [-1, 116], [-1, 108], [-1, 101], [-1, 62], [-1, 67], [-1, 111], [-1, 109], [-1, 112], [-1, 114], [-1, 101], [-1, 115], [-1, 115], [-1, 105], [-1, 111], [-1, 110], [-1, 32], [-1, 80], [-1, 111], [-1, 105], [-1, 110], [-1, 116], [-1, 101], [-1, 114], [-1, 115], [-1, 60], [-1, 47], [8, 6], [-1, 10], [-1, 60], [-1, 77], [-1, 69], [-1, 84], [-1, 65], [-1, 32], [-1, 72]]}
```

### Computing minimum SLP grammar with PySAT

The following program will compute minimum SLP grammar of size `factor_size = 68` for `cp.html-50` of length 50.

```bash
pipenv run python src/slp_solver.py --file data/cantrbry_pref/cp.html-50
{"date": "2022-04-19 20:55:49.267226", "status": "", "algo": "slp-sat", "file_name": "cp.html-50", "file_len": 50, "time_prep": 0.05850052833557129, "time_total": 0.06189441680908203, "sol_nvars": 2693, "sol_nhard": 28706, "sol_nsoft": 50, "sol_navgclause": 2.7268166933742073, "sol_ntotalvars": 78276, "sol_nmaxclause": 52, "factor_size": 68, "factors": []}
```
