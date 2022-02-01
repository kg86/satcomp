# Run a benchmark for LZ77, LZRR, lcpcomp, lexparse
# We use the following program by @TNishimoto
# https://github.com/TNishimoto/lzrr

import argparse
from dataclasses import asdict, dataclass
import datetime
import glob
import sys
import time
from joblib import Parallel, delayed
import sqlite3
import subprocess
import os


dbname = "out/satcomp.db"
dbtable = "lz_bench"

files = (
    glob.glob("data/calgary_pref/*-50")
    + glob.glob("data/calgary_pref/*-100")
    + glob.glob("data/cantrbry_pref/*-50")
    + glob.glob("data/cantrbry_pref/*-100")
)
files = [os.path.abspath(f) for f in files]

algos = ["lz", "lzrr", "lex", "lcp"]


@dataclass
class LZExp:
    date: str
    status: str
    algo: str
    file_name: str
    file_len: int
    time_total: float
    factor_size: int

    @classmethod
    def create(cls):
        return LZExp("", "", "", "", 0, 0, 0)


def benchmark_program(timeout, algo, file) -> list[str]:
    """
    runs program with given setting (timeout, algo, file).
    """
    base_name = os.path.basename(file)
    out_file = f"out/lz/{base_name}.{algo}"
    cmd = [
        "externals/lzrr/build/compress.out",
        "--input_file",
        file,
        "--output_file",
        out_file,
        "--mode",
        algo,
    ]
    print(" ".join(cmd))
    num_factor = 0
    time_start = time.time()
    try:
        out = subprocess.check_output(cmd, shell=False, timeout=timeout)
        res_pre = b"The number of factors : "
        res_beg = out.find(res_pre) + len(res_pre)
        res_end = out.find(b"\n", res_beg)
        num_factor = int(out[res_beg:res_end])
        status = "complete"
    except subprocess.TimeoutExpired:
        status = f"timeout-{timeout}"
    except Exception:
        status = "error"

    exp = LZExp(
        str(datetime.datetime.now()),
        status,
        algo,
        os.path.basename(file),
        len(open(file, "rb").read()),
        time.time() - time_start,
        num_factor,
    )

    expd = exp.__dict__
    return list(map(str, expd.values()))


def benchmark_mul(timeout, algos, files, out_file, n_jobs):
    """
    run benchmark program with multiple processes
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
    expd = LZExp.create().__dict__  # asdict dowsn't work, I don't know the reason
    n = len(expd.keys())
    cur.executemany(
        f"INSERT INTO {dbtable} VALUES ({', '.join('?' for _ in range(n))})",
        queries,  # type: ignore
    )
    con.commit()


def clear_table(table_name):
    """
    delete table if exists, and create new table.
    """
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    exp = LZExp.create()
    expd = asdict(exp)

    try:
        cur.execute(f"DROP TABLE {table_name}")
    except sqlite3.OperationalError:
        pass
    cur.execute(f"CREATE TABLE {table_name} ({', '.join(key for key in expd.keys())})")
    con.commit()


def export_csv(table_name, out_file):
    """
    export table to csv.
    """
    con = sqlite3.connect(dbname)
    import pandas as pd

    df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
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
    clear_table(dbtable)
    args = parse_args()
    benchmark_mul(args.timeout, algos, files, args.output, args.n_jobs)
    export_csv(dbtable, args.output)


if __name__ == "__main__":
    main()
