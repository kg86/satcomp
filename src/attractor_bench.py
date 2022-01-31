import argparse
import datetime
import glob
import sys
from joblib import Parallel, delayed
import sqlite3
import subprocess
import os

from attractor import AttractorType, verify_attractor
from attractor_bench_format import AttractorExp

dbname = "attractor_bench.db"

files = (
    glob.glob("data/calgary_pref/*-50")
    + glob.glob("data/calgary_pref/*-100")
    + glob.glob("data/cantrbry_pref/*-50")
    + glob.glob("data/cantrbry_pref/*-100")
)
files = [os.path.abspath(f) for f in files]

algos = ["solver"]


def run_solver(input_file: str, timeout: float = None) -> AttractorExp:
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
        valid = verify_attractor(open(input_file, "rb").read(), exp.attractor)
        exp.status = "correct" if valid else "wrong"
        status = "complete"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception:
        status = "error"

    print(f"status: {status}")
    if status != "complete":
        exp = AttractorExp(
            str(datetime.datetime.now()),
            status,
            "solver",
            os.path.basename(input_file),
            len(open(input_file, "rb").read()),
            0,
            0,
            0,
            0,
            0,
            0,
            AttractorType([]),
        )
    assert isinstance(exp, AttractorExp)
    return exp


def benchmark_program(timeout, algo, file, out_file):
    """
    runs program with given setting (timeout, algo, file).
    """
    if algo == "solver":
        exp = run_solver(file, timeout)
    else:
        assert False
    with open(out_file, "a") as f:
        f.write(exp.to_json(ensure_ascii=False) + "\n")  # type: ignore

    con = sqlite3.connect(dbname)
    cur = con.cursor()
    json = exp.to_dict()  # type: ignore
    n = len(json.values())
    cur.execute(
        f"INSERT INTO attr_bench VALUES ({', '.join('?' for _ in range(n))})",
        tuple(map(str, json.values())),
    )
    con.commit()


def benchmark_mul(timeout, algos, files, out_file, n_jobs):
    """
    run benchmark program with multiple processes
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


def clear_table():
    """
    delete table if exists, and create new table.
    """
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    exp = AttractorExp.create()
    d = exp.to_dict()  # type: ignore

    cur.execute("DROP TABLE attr_bench")
    cur.execute(f"CREATE TABLE attr_bench ({', '.join(key for key in d.keys())})")


def export_csv(out_file):
    """
    export table to csv.
    """
    con = sqlite3.connect(dbname)
    import pandas as pd

    df = pd.read_sql_query("SELECT * FROM attr_bench", con)
    df.to_csv(out_file, index=False)


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
    clear_table()
    args = parse_args()
    benchmark_mul(args.timeout, algos, files, args.output, args.n_jobs)
    export_csv(args.output)


if __name__ == "__main__":
    main()
