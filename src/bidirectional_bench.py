import argparse
import datetime
import os
import sqlite3
import subprocess
import sys
import time
from typing import List, Optional

from joblib import Parallel, delayed

import bidirectional
from bidirectional import BiDirExp, BiDirType

dbname = "out/satcomp.db"
dbtable = "bidirectional_bench"

algos = ["solver"]


def run_naive(input_file: str, timeout: Optional[float] = None) -> BiDirExp:
    input_file = os.path.abspath(input_file)
    current_dir = os.path.abspath(".")
    os.chdir("rust")
    cmd = ["cargo", "run", "--bin", "optimal_bms", "--", "--input_file", input_file]
    print(" ".join(cmd))
    start = time.time()
    out = None
    try:
        out = subprocess.check_output(
            cmd, shell=False, timeout=timeout, stderr=subprocess.DEVNULL
        )
        status = "complete"
    except subprocess.TimeoutExpired:
        print("timeout")
        status = f"timeout-{timeout}"
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
        date=str(datetime.datetime.now()),
        status=status,
        algo="naive",
        file_name=os.path.basename(input_file),
        file_len=len(open(input_file, "rb").read()),
        time_prep=0,
        time_total=time_total,
        sol_nvars=0,
        sol_nhard=0,
        sol_nsoft=0,
        factor_size=len(bd),
        sol_navgclause=0.0,
        sol_ntotalvars=0,
        sol_nmaxclause=0,
        factors=bd,
    )


def run_solver(input_file: str, timeout: Optional[float] = None) -> BiDirExp:
    cmd = [
        "pipenv",
        "run",
        "python",
        "src/bidirectional_solver.py",
        "--file",
        input_file,
    ]
    print(" ".join(cmd))
    # start = time.time()
    exp = None
    try:
        out = subprocess.check_output(cmd, shell=False, timeout=timeout)
        last1 = out.rfind(b"\n")
        last2 = out.rfind(b"\n", 0, last1)
        print(out[last2 + 1 : last1])
        exp = BiDirExp.from_json(out[last2 + 1 : last1])  # type: ignore
        exp.status = "complete"
        status = "complete"
    except subprocess.TimeoutExpired:
        status = f"timeout-{timeout}"
    except Exception:
        status = "error"

    print(f"status: {status}")
    if status == "complete":
        assert exp
    else:
        exp = BiDirExp(
            date=str(datetime.datetime.now()),
            status=status,
            algo="solver",
            file_name=os.path.basename(input_file),
            file_len=len(open(input_file, "rb").read()),
            time_prep=0,
            time_total=0,
            sol_nvars=0,
            sol_nhard=0,
            sol_nsoft=0,
            factor_size=0,
            sol_navgclause=0.0,
            sol_ntotalvars=0,
            sol_nmaxclause=0,
            factors=BiDirType([]),
        )
    return exp


def benchmark_program(timeout, algo, file) -> List[str]:
    """
    Run program with given setting (timeout, algo, file).
    """
    if algo == "naive":
        exp = run_naive(file, timeout)
    elif algo == "solver":
        exp = run_solver(file, timeout)
    else:
        assert False

    # verify the result
    if exp.status == "complete":
        if bidirectional.decode(exp.factors) == open(file, "rb").read():
            exp.status = "correct"
        else:
            exp.status = "wrong"
    expd = exp.__dict__
    return list(map(str, expd.values()))


def benchmark_single(timeout, algos, files, out_file):
    """
    Run program with single process.
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
    Run programs with multiple processes.
    """
    if os.path.exists(out_file):
        os.remove(out_file)
    queries = Parallel(n_jobs=n_jobs)(
        [
            delayed(benchmark_program)(timeout, algo, file)
            for file in files
            for algo in algos
        ]
    )
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    expd = BiDirExp.create().__dict__  # asdict dowsn't work, I don't know the reason
    n = len(expd.keys())
    cur.executemany(
        f"INSERT INTO {dbtable} VALUES ({', '.join('?' for _ in range(n))})",
        queries,  # type: ignore
    )
    con.commit()


def clear_table(dbtable):
    """
    Delete table if exists, and create new table.
    """
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    exp = BiDirExp.create()

    expd = exp.__dict__

    try:
        cur.execute(f"DROP TABLE {dbtable}")
    except sqlite3.OperationalError:
        pass
    cur.execute(f"CREATE TABLE {dbtable} ({', '.join(key for key in expd.keys())})")


def export_csv(dbtable, out_file):
    """
    Export table as csv format.
    """
    con = sqlite3.connect(dbname)
    import pandas as pd

    df = pd.read_sql_query(f"SELECT * FROM {dbtable}", con)
    df.to_csv(out_file, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark for algorithms computing the smallest bidirectional scheme."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="timeout (sec). If 0 is set, the proguram does not time out.",
        default=60,
    )
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument("--n_jobs", type=int, help="number of jobs", default=2)
    parser.add_argument("--files", nargs="*", help="files", default=[])

    args = parser.parse_args()
    if args.output == "" or args.timeout < 0 or len(args.files) == 0:
        parser.print_help()
        sys.exit()
    if args.timeout == 0:
        args.timeout = None
    return args


def main():
    clear_table(dbtable)
    args = parse_args()
    benchmark_mul(args.timeout, algos, args.files, args.output, args.n_jobs)
    export_csv(dbtable, args.output)


if __name__ == "__main__":
    main()
