import datetime
import subprocess
import glob
import os
from dataclasses import dataclass
import time
from joblib import Parallel, delayed


from bidirectional import BiDirExp, BiDirType
from bidirectional_solver import bidirectional, bidirectional_hdbn

exp_file_tmplate = "exp_timeout={}.txt"

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
    cmd = f"cd rustr-master && cargo run --bin optimal_bms -- --input_file {input_file}"
    print(cmd)
    start = time.time()
    out = None
    try:
        out = subprocess.check_output(
            cmd, shell=True, timeout=timeout, stderr=subprocess.DEVNULL
        )
        status = "complete"
    except subprocess.TimeoutExpired:
        print(f"timeout")
        status = "timeout"
    except Exception:
        status = "error"

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
    cmd = f"pipenv run python bidirectional_solver.py  --file {input_file}"
    print(cmd)
    start = time.time()
    exp = None
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=timeout)
        # out = subprocess.check_output(
        #     cmd, shell=True, stderr=subprocess.DEVNULL, timeout=timeout
        # )
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


def benchmark_program(timeout, algo, file):
    """
    runs program with given setting (timeout, algo, file).
    """
    if algo == "naive":
        exp = run_naive(file, timeout)
    elif algo == "solver":
        exp = run_solver(file, timeout)
    else:
        assert False
    with open(exp_file_tmplate.format(timeout), "a") as f:
        f.write(exp.to_json(ensure_ascii=False) + "\n")  # type: ignore


def benchmark_single(timeout, algos, files):
    """
    runs program with single process.
    """
    for file in files:
        for algo in algos:
            if algo == "naive":
                exp = run_naive(file, timeout)
            elif algo == "solver":
                exp = run_solver(file, timeout)
            else:
                assert False
            out.write(exp.to_json(ensure_ascii=False) + "\n")  # type: ignore


def benchmark_mul(timeout, algos, files):
    """
    runs programs with multiple processes.
    """
    exp_file = exp_file_tmplate.format(timeout)
    if os.path.exists(exp_file):
        os.remove(exp_file)
    result = Parallel(n_jobs=4)(
        [
            delayed(benchmark_program)(timeout, algo, file)
            for file in files
            for algo in algos
        ]
    )


def main():
    timeout = 60
    benchmark_mul(timeout, algos, files)
    # benchmark_single(timeout, algos, files)


if __name__ == "__main__":
    main()
