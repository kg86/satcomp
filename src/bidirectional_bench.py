import argparse
import datetime
import subprocess
import glob
import os
import sys
from dataclasses import dataclass
import time
from joblib import Parallel, delayed


from bidirectional import BiDirExp, BiDirType


files = (
    glob.glob("data/calgary_pref/*-50")
    + glob.glob("data/calgary_pref/*-100")
    + glob.glob("data/cantrbry_pref/*-50")
    + glob.glob("data/cantrbry_pref/*-100")
)
files = [os.path.abspath(f) for f in files]

algos = ["solver", "naive"]


def run_naive(input_file: str, timeout: float = None) -> BiDirExp:
    input_file = os.path.abspath(input_file)
    current_dir = os.path.abspath(".")
    os.chdir("rustr-master")
    cmd = ["cargo", "run", "--bin", "optimal_bms", "--", "--input_file", input_file]
    print(cmd)
    start = time.time()
    out = None
    try:
        out = subprocess.check_output(
            cmd, shell=False, timeout=timeout, stderr=subprocess.DEVNULL
        )
        status = "complete"
    except subprocess.TimeoutExpired:
        print(f"timeout")
        status = "timeout"
    except Exception:
        status = "error"
    os.chdir(current_dir)

    print(f"status: {status}")
    if status == "complete":
        assert out
        time_total = time.time() - start
        last1 = out.rfind(b"\n")
        last2 = out.rfind(b"\n", 0, last1)
        bd = eval(out[last2:last1])
    else:
        time_total = 0
        bd = BiDirType([])

    return BiDirExp(
        str(datetime.datetime.now()),
        status,
        "naive",
        os.path.basename(input_file),
        len(open(input_file, "rb").read()),
        0,
        time_total,
        len(bd),
        bd,
        0,
        0,
        0,
    )


def run_solver(input_file: str, timeout: float = None) -> BiDirExp:
    cmd = ["pipenv", "run", "python", "bidirectional_solver.py", "--file", input_file]
    print(cmd)
    start = time.time()
    exp = None
    try:
        out = subprocess.check_output(cmd, shell=False, timeout=timeout)
        last1 = out.rfind(b"\n")
        last2 = out.rfind(b"\n", 0, last1)
        print(out[last2 + 1 : last1])
        exp = BiDirExp.from_json(out[last2 + 1 : last1])  # type: ignore
        exp.status = "complete"
        time_total = time.time() - start
        status = "complete"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception:
        status = "error"

    print(f"status: {status}")
    if status == "complete":
        assert exp
    else:
        exp = BiDirExp(
            str(datetime.datetime.now()),
            status,
            "solver",
            os.path.basename(input_file),
            len(open(input_file, "rb").read()),
            0,
            0,
            0,
            BiDirType([]),
            0,
            0,
            0,
        )
    return exp


def benchmark_program(timeout, algo, file, out_file):
    """
    runs program with given setting (timeout, algo, file).
    """
    if algo == "naive":
        exp = run_naive(file, timeout)
    elif algo == "solver":
        exp = run_solver(file, timeout)
    else:
        assert False
    with open(out_file, "a") as f:
        f.write(exp.to_json(ensure_ascii=False) + "\n")  # type: ignore


def benchmark_single(timeout, algos, files, out_file):
    """
    runs program with single process.
    """
    f = open(out_file, "w")
    for file in files:
        for algo in algos:
            if algo == "naive":
                exp = run_naive(file, timeout)
            elif algo == "solver":
                exp = run_solver(file, timeout)
            else:
                assert False
            f.write(exp.to_json(ensure_ascii=False) + "\n")  # type: ignore


def benchmark_mul(timeout, algos, files, out_file, n_jobs):
    """
    runs programs with multiple processes.
    """
    if os.path.exists(out_file):
        os.remove(out_file)
    result = Parallel(n_jobs=n_jobs)(
        [
            delayed(benchmark_program)(timeout, algo, file, out_file)
            for file in files
            for algo in algos
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark for algorithms computing the smallest bidirectional scheme"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="timeout (sec). If 0 is set, the proguram does not timeout.",
        default=60,
    )
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument("--n_jobs", type=int, help="number of jobs", default=2)

    args = parser.parse_args()
    if args.output == "" or args.timeout < 0:
        parser.print_help()
        sys.exit()
    if args.timeout == 0:
        args.timeout = None
    return args


def main():
    args = parse_args()
    benchmark_mul(args.timeout, algos, files, args.output, args.n_jobs)
    # benchmark_single(timeout, algos, files)


if __name__ == "__main__":
    main()
