#!/bin/bash
# timeout=3600
timeout=6
jobs=4
files="data/calgary/* data/misc/fib0[0-9].txt data/misc/pds* data/misc/thuemorse* data/misc/trib*"

uv run src/lz_bench.py --timeout=${timeout} --output=out/lz_bnech.csv --n_jobs $jobs --files $files
uv run src/attractor_bench.py --timeout=${timeout} --output=out/attractor_bench.csv --n_jobs $jobs --files $files
uv run src/bidirectional_bench.py --timeout=${timeout} --output=out/bidirectional_bench.csv --n_jobs $jobs --files $files

uv run src/benchmark_common.py
