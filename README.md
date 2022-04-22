# Computing NP-hard Repetitiveness Measure by SAT solvers

This software package contains several Python scripts for computing
- a smallest smallest string attractor,
- a smallest straight-line program, and
- a bidirectional macro scheme with the fewest phrases,
by using the SAT solver pySAT.

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

1. Install `pipenv` and `python3.8` on your OS.
2. Run the following command at the root of this repository.

```console
pipenv sync
```
This installs package dependencies like `pysat` locally to the repository, which are needed for running the Python scripts.
If the command fails, it is likely that you have a different minor version of `python3` installed.
In most of the cases, you can exchange the line `python_version` in `Pipfile` with your Python version.

## Usage

Our executables are written in Python, and can be accessed from the `src` directory.
You can run the programs via the `pipenv` command as follows.

```console
pipenv run python src/slp_solver.py --str "abracadabra"
pipenv run python src/attractor_solver.py --algo min --str "abracadabra"
pipenv run python src/bidirectional_solver.py --str "abracadabra"
```
Common to all Python scripts are the input parameters `--str` for giving a string as an input, or `--file` for reading an input file.


With parameter `-h` you obtain the list of all available parameters:
```console
pipenv run python src/attractor_solver.py -h
```
```
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

## Output format

The output consists of, omitting some logging information, a line of JSON,
which stores the following attributes:

- `date`: the date of execution
- `status`: empty in case of success; otherwise it can be used for error messages
- `algo`: the name of the used program like `slp-sat` for our solution computing the smallest SLP
- `file_name`: the name of the input file (empty if `--str` is used)
- `file_len`: the length of the input file (0 if `--str` is used)
- `time_prep`: the time needed for defining all hard and soft clauses
- `time_total`: the total running time
- `sol_nvars`: number of defined Boolean variables for the solver
- `sol_nhard`: number of defined hard clauses
- `sol_nsoft`: number of defined soft clauses
- `sol_navgclause`: average number of literals a clause contains
- `sol_ntotalvars`: the size of the CNF, i.e., the sum of all literals in each hard clauses
- `sol_nmaxclause`: the number of literals in the largest clause
- `factor_size`: the size of the output (i.e., the minimal number of factors of a BMS, the minimal size of a string attractor, the mimimal size of a grammar)
- `factors`: an instance of a valid output attaining the size `factor_size`. This is
  * for string attractors a list of text positions
  * for BMS a list of pairs [pos, len] for the direction to copy from `T[pos]` a substring of length `len`. If `pos` is -1, then the factor is a ground phrase and it stores its character in `len`.
  * for SLP a list of non-terminals specified in the form 
      1) (`from`, `to`, `char`): [], or
      2) (`from`, `to`, None): [(`fromLeft`, `toLeft`, `charLeft`), (`fromRight`, `toRight`, `charRight`)]

    In both cases, it defines a non-terminal whose expansion is the substring T[`from`..`to`-1].
    In the first case, we know that this non-terminal expands to a single character `char`.
    In the second case, the non-terminal has two non-terminals children, each defined again by this triplet of `from`, `to`, and `char` value.

Please see below for concrete examples in how the output looks like.

## Running Examples

### Computing minimum string attractors with PySAT

The following program compute a minimum string attractor of size `factor_size = 497` for the dataset `grammar.lsp` having a length of 3721.

```bash
pipenv run python src/attractor_solver.py --file data/cantrbry/grammar.lsp --algo min
```
```json
{"date": "2022-04-19 20:42:38.553750", "status": "", "algo": "attractor-sat", "file_name": "grammar.lsp", "file_len": 3721, "time_prep": 0.08829855918884277, "time_total": 0.38588380813598633, "sol_nvars": 3721, "sol_nhard": 1669, "sol_nsoft": 3721, "sol_navgclause": 17.860395446375076, "sol_ntotalvars": 29809, "sol_nmaxclause": 802, "factor_size": 497, "factors": [2, 5, 8, 10, 12, 14, 20, 22, 25, 28, 31, 34, 38, 44, 50, 51, 54, 57, 59, 62, 65, 69, 72, 77, 80, 85, 95, 99, 108, 114, 118, 121, 125, 133, 139, 143, 155, 161, 170, 181, 183, 186, 194, 201, 226, 239, 255, 261, 268, 276, 289, 294, 298, 307, 310, 313, 326, 341, 346, 356, 358, 378, 411, 429, 440, 449, 454, 472, 482, 489, 510, 523, 535, 546, 591, 596, 628, 650, 675, 677, 688, 707, 719, 725, 741, 759, 766, 785, 796, 805, 829, 845, 854, 872, 913, 917, 930, 941, 959, 981, 1005, 1011, 1018, 1029, 1031, 1051, 1053, 1064, 1082, 1103, 1108, 1118, 1129, 1149, 1167, 1176, 1186, 1189, 1216, 1224, 1234, 1253, 1272, 1275, 1280, 1286, 1292, 1294, 1300, 1305, 1316, 1321, 1327, 1335, 1346, 1354, 1369, 1377, 1400, 1413, 1430, 1432, 1435, 1482, 1498, 1512, 1552, 1573, 1584, 1585, 1587, 1591, 1613, 1629, 1631, 1644, 1670, 1717, 1732, 1738, 1742, 1744, 1746, 1749, 1752, 1754, 1762, 1770, 1776, 1782, 1786, 1788, 1792, 1795, 1798, 1804, 1806, 1809, 1811, 1815, 1819, 1821, 1825, 1831, 1841, 1851, 1857, 1862, 1868, 1874, 1880, 1885, 1891, 1898, 1901, 1904, 1913, 1915, 1918, 1925, 1928, 1933, 1942, 1952, 1956, 1962, 1964, 1966, 1972, 1978, 1983, 2000, 2005, 2013, 2017, 2022, 2028, 2031, 2040, 2044, 2056, 2062, 2074, 2076, 2085, 2089, 2101, 2106, 2109, 2112, 2119, 2123, 2124, 2133, 2143, 2147, 2150, 2153, 2157, 2159, 2165, 2168, 2175, 2177, 2178, 2183, 2190, 2197, 2206, 2212, 2214, 2216, 2220, 2224, 2226, 2228, 2230, 2239, 2243, 2248, 2252, 2256, 2261, 2266, 2277, 2281, 2290, 2295, 2299, 2303, 2309, 2312, 2316, 2321, 2335, 2340, 2344, 2360, 2362, 2366, 2379, 2383, 2386, 2401, 2404, 2418, 2428, 2448, 2457, 2470, 2472, 2491, 2496, 2500, 2516, 2524, 2536, 2540, 2558, 2566, 2569, 2572, 2584, 2587, 2601, 2613, 2622, 2627, 2631, 2639, 2649, 2657, 2664, 2680, 2690, 2693, 2696, 2705, 2718, 2721, 2741, 2745, 2749, 2762, 2766, 2777, 2793, 2797, 2808, 2815, 2825, 2836, 2840, 2844, 2859, 2868, 2871, 2894, 2900, 2902, 2916, 2918, 2921, 2927, 2943, 2964, 2968, 2974, 2990, 2993, 2998, 3004, 3017, 3023, 3026, 3037, 3051, 3056, 3065, 3068, 3070, 3074, 3085, 3092, 3099, 3103, 3115, 3119, 3122, 3131, 3139, 3142, 3147, 3150, 3152, 3159, 3164, 3165, 3177, 3184, 3197, 3202, 3205, 3209, 3216, 3220, 3222, 3230, 3235, 3241, 3249, 3252, 3257, 3259, 3264, 3268, 3271, 3278, 3281, 3288, 3295, 3298, 3301, 3305, 3314, 3319, 3324, 3328, 3335, 3336, 3348, 3360, 3366, 3369, 3374, 3377, 3385, 3391, 3394, 3398, 3405, 3414, 3420, 3422, 3428, 3431, 3440, 3443, 3449, 3454, 3458, 3465, 3467, 3471, 3474, 3477, 3480, 3483, 3486, 3490, 3493, 3500, 3504, 3515, 3519, 3521, 3526, 3530, 3534, 3544, 3550, 3555, 3558, 3560, 3564, 3570, 3573, 3579, 3581, 3585, 3590, 3592, 3594, 3598, 3603, 3606, 3609, 3612, 3624, 3626, 3627, 3631, 3634, 3636, 3639, 3642, 3644, 3645, 3647, 3649, 3651, 3653, 3655, 3662, 3668, 3671, 3676, 3677, 3681, 3685, 3689, 3692, 3694, 3697, 3701, 3705, 3712]}
```

### Computing minimum bidirectional macroschemes with PySAT

The following program will compute minimum bidirectional macroschemes of size `factor_size = 43` for `cp.html-50` of length 50.

```bash
pipenv run python src/bidirectional_solver.py --file data/cantrbry_pref/cp.html-50 
```
```json
{"date": "2022-04-19 20:51:17.510793", "status": "", "algo": "bidirectional-sat", "file_name": "cp.html-50", "file_len": 50, "time_prep": 0.010410785675048828, "time_total": 0.014687776565551758, "sol_nvars": 1140, "sol_nhard": 3517, "sol_nsoft": 50, "sol_navgclause": 2.3810065396644866, "sol_ntotalvars": 8374, "sol_nmaxclause": 761, "factor_size": 43, "factors": [[-1, 60], [-1, 104], [-1, 101], [-1, 97], [-1, 100], [41, 3], [-1, 116], [-1, 105], [-1, 116], [-1, 108], [-1, 101], [-1, 62], [-1, 67], [-1, 111], [-1, 109], [-1, 112], [-1, 114], [-1, 101], [-1, 115], [-1, 115], [-1, 105], [-1, 111], [-1, 110], [-1, 32], [-1, 80], [-1, 111], [-1, 105], [-1, 110], [-1, 116], [-1, 101], [-1, 114], [-1, 115], [-1, 60], [-1, 47], [8, 6], [-1, 10], [-1, 60], [-1, 77], [-1, 69], [-1, 84], [-1, 65], [-1, 32], [-1, 72]]}
```

### Computing minimum SLP grammar with PySAT

The following program will compute minimum SLP grammar of size `factor_size = 68` for `cp.html-50` of length 50.

```console
pipenv run python src/slp_solver.py --file data/cantrbry_pref/cp.html-50
```
```json
{"date": "2022-04-22 12:20:16.308535", "status": "", "algo": "slp-sat", "file_name": "cp.html-50", "file_len": 50, "time_prep": 0.05077958106994629, "time_total": 0.053725481033325195, "sol_nvars": 2693, "sol_nhard": 28704, "sol_nsoft": 50, "sol_navgclause": 2.7269370122630994, "sol_ntotalvars": 78274, "sol_nmaxclause": 52, "factor_size": 68, "factors": "{(49, 50, 72): [], (48, 49, 32): [], (47, 48, 65): [], (46, 47, 84): [], (45, 46, 69): [], (44, 45, 77): [], (42, 44, 6): [], (36, 42, 8): [], (35, 36, 47): [], (34, 35, 60): [], (33, 34, 115): [], (32, 33, 114): [], (31, 32, 101): [], (30, 31, 116): [], (29, 30, 110): [], (28, 29, 105): [], (27, 28, 111): [], (26, 27, 80): [], (25, 26, 32): [], (24, 25, 110): [], (23, 24, 111): [], (22, 23, 105): [], (21, 22, 115): [], (20, 21, 115): [], (19, 20, 101): [], (18, 19, 114): [], (17, 18, 112): [], (16, 17, 109): [], (15, 16, 111): [], (14, 15, 67): [], (13, 14, 62): [], (12, 13, 101): [], (11, 12, 108): [], (10, 11, 116): [], (9, 10, 105): [], (8, 9, 116): [], (8, 14, None): [(8, 13, None), (13, 14, 62)], (7, 8, 60): [], (6, 7, 10): [], (6, 8, None): [(6, 7, 10), (7, 8, 60)], (5, 6, 62): [], (4, 5, 100): [], (3, 4, 97): [], (2, 3, 101): [], (1, 2, 104): [], (0, 1, 60): [], (0, 50, None): [(0, 49, None), (49, 50, 72)], (0, 2, None): [(0, 1, 60), (1, 2, 104)], (0, 3, None): [(0, 2, None), (2, 3, 101)], (0, 4, None): [(0, 3, None), (3, 4, 97)], (0, 5, None): [(0, 4, None), (4, 5, 100)], (0, 6, None): [(0, 5, None), (5, 6, 62)], (0, 8, None): [(0, 6, None), (6, 8, None)], (0, 14, None): [(0, 8, None), (8, 14, None)], (0, 15, None): [(0, 14, None), (14, 15, 67)], (0, 16, None): [(0, 15, None), (15, 16, 111)], (0, 17, None): [(0, 16, None), (16, 17, 109)], (0, 18, None): [(0, 17, None), (17, 18, 112)], (0, 19, None): [(0, 18, None), (18, 19, 114)], (0, 20, None): [(0, 19, None), (19, 20, 101)], (0, 21, None): [(0, 20, None), (20, 21, 115)], (0, 22, None): [(0, 21, None), (21, 22, 115)], (0, 23, None): [(0, 22, None), (22, 23, 105)], (0, 24, None): [(0, 23, None), (23, 24, 111)], (0, 25, None): [(0, 24, None), (24, 25, 110)], (0, 26, None): [(0, 25, None), (25, 26, 32)], (0, 27, None): [(0, 26, None), (26, 27, 80)], (0, 28, None): [(0, 27, None), (27, 28, 111)], (0, 29, None): [(0, 28, None), (28, 29, 105)], (0, 30, None): [(0, 29, None), (29, 30, 110)], (0, 31, None): [(0, 30, None), (30, 31, 116)], (0, 32, None): [(0, 31, None), (31, 32, 101)], (0, 33, None): [(0, 32, None), (32, 33, 114)], (0, 34, None): [(0, 33, None), (33, 34, 115)], (0, 35, None): [(0, 34, None), (34, 35, 60)], (0, 36, None): [(0, 35, None), (35, 36, 47)], (0, 42, None): [(0, 36, None), (36, 42, 8)], (0, 44, None): [(0, 42, None), (42, 44, 6)], (0, 45, None): [(0, 44, None), (44, 45, 77)], (0, 46, None): [(0, 45, None), (45, 46, 69)], (0, 47, None): [(0, 46, None), (46, 47, 84)], (0, 48, None): [(0, 47, None), (47, 48, 65)], (0, 49, None): [(0, 48, None), (48, 49, 32)], (8, 10, None): [(8, 9, 116), (9, 10, 105)], (8, 11, None): [(8, 10, None), (10, 11, 116)], (8, 12, None): [(8, 11, None), (11, 12, 108)], (8, 13, None): [(8, 12, None), (12, 13, 101)]}"}
```

### Evaluation of Test Datasets

We have a collection of test datasets in the folder `data`.
To measure the output sizes of these datasets, we provide the script `shell/measure_datasets.sh`.
It requires the program `jq` to be installed, and outputs the JSON file `shell/measure/stats.json` storing for each file the output size of each computed compression measure.
If you apply this script on a SLURM cluster, then it batches the experiments. Collecting the final data has to be done manually (the last lines in the shell script).

### Notes

The code has been tested only on Linux.
