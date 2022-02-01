#!/bin/bash
timeout=30
pipenv run python src/lz_bench.py --timeout=${timeout} --output=out/lz_bnech.csv --n_jobs 8
pipenv run python src/attractor_bench.py --timeout=${timeout} --output=out/attractor_bench.csv --n_jobs 8
pipenv run python src/bidirectional_bench.py --timeout=${timeout} --output=out/bidirectional_bench.csv --n_jobs 8

pipenv run python src/benchmark_common.py