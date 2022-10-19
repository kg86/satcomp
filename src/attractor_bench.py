import argparse
import datetime
import glob
import sys
from typing import Optional
from joblib import Parallel, delayed
import sqlite3
import subprocess
import os

from attractor import AttractorType, verify_attractor
from attractor_bench_format import AttractorExp

dbname = "out/satcomp.db"
dbtable = "attractor_bench"


algos = ["solver"]


def run_solver(input_file: str, timeout: Optional[float] = None) -> AttractorExp:
    cmd = [
        "pipenv",
        "run",
        "python",
        "src/attractor_solver.py",
        "--algo",
        "min",
        "--file",
        input_file,
    ]
    print(" ".join(cmd))
    exp = None
    try:
        out = subprocess.check_output(cmd, shell=False, timeout=timeout)
        last1 = out.rfind(b"\n")
        last2 = out.rfind(b"\n", 0, last1)
        print(out[last2 + 1 : last1])
        exp = AttractorExp.from_json(out[last2 + 1 : last1])  # type: ignore
        status = "complete"
        exp.status = status
    except subprocess.TimeoutExpired:
        status = f"timeout-{timeout}"
    except Exception:
        print("Unexpected error:", sys.exc_info()[0])
        status = "error"

    print(f"status: {status}")
    if status != "complete":
        exp = AttractorExp(
                is_satisfied =            False,
                is_optimal =            False,
                date =                  str(datetime.datetime.now()),
                status =                status,
                algo =                  "solver",
                file_name =             os.path.basename(input_file),
                file_len =              len(open(input_file, "rb").read()),
                time_prep =             0,
                time_total =            0,
                sol_nvars =             0,
                sol_nhard =             0,
                sol_nsoft =             0,
                factor_size =           0,
                sol_navgclause = 0,
                sol_ntotalvars = 0,
                sol_nmaxclause = 0,
                factors =               AttractorType([]),
                )
        assert isinstance(exp, AttractorExp)
    return exp


def benchmark_program(timeout, algo, file):
    """
    Runs program with given setting (timeout, algo, file).
    """
    if algo == "solver":
        exp = run_solver(file, timeout)
    else:
        assert False

    # verify the result
    if exp.status == "complete":
        valid = verify_attractor(open(file, "rb").read(), exp.factors)
        exp.status = "correct" if valid else "wrong"

    con = sqlite3.connect(dbname)
    cur = con.cursor()
    json = exp.to_dict()  # type: ignore
    n = len(json.values())
    cur.execute(
        f"INSERT INTO {dbtable} VALUES ({', '.join('?' for _ in range(n))})",
        tuple(map(str, json.values())),
    )
    con.commit()


def benchmark_mul(timeout, algos, files, n_jobs):
    """
    Run benchmark program with multiple processes
    """
    Parallel(n_jobs=n_jobs)(
        [
            delayed(benchmark_program)(timeout, algo, file)
            for file in files
            for algo in algos
        ]
    )


def clear_table():
    """
    Delete table if exists, and create new table.
    """
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    exp = AttractorExp.create()
    d = exp.to_dict()  # type: ignore

    try:
        cur.execute(f"DROP TABLE {dbtable}")
    except sqlite3.OperationalError:
        pass
    cur.execute(f"CREATE TABLE {dbtable} ({', '.join(key for key in d.keys())})")


def export_csv(out_file):
    """
    Store table as csv format in `out_file`.
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
        help="timeout (sec). If 0 is set, the program does not time out",
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
    clear_table()
    args = parse_args()
    benchmark_mul(args.timeout, algos, args.files, args.n_jobs)
    export_csv(args.output)


if __name__ == "__main__":
    main()
