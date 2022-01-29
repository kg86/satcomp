import argparse
import csv
import datetime
import glob
import sys
import time
from typing import Iterable
from joblib import Parallel, delayed
import sqlite3
import tempfile
import subprocess
import os

from attractor import AttractorType
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


def gen(n: int) -> Iterable[str]:
    """
    Compute all binary strings of length `n`.
    """
    if n == 1:
        yield "a"
        yield "b"
    elif n > 1:
        for suf in gen(n - 1):
            yield suf + "a"
            yield suf + "b"
    else:
        assert False


prog_msa = "to_wcnf"
prog_solver = "open-wbo"


def num_pos(xs: list[int]) -> int:
    return sum(1 for x in xs if x > 0)


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
        exp.status = "complete"
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
            AttractorType([]),
            0,
            0,
            0,
        )
    assert isinstance(exp, AttractorExp)
    return exp


def run_solver_other(x: str):
    # make file
    input_file = f"exp/tmp/{x}.txt"
    open(input_file, "wb").write(x.encode("utf8"))
    cmd_wcnf = ["to_wcnf", "-i", input_file]
    subprocess.check_call(cmd_wcnf)

    # run solver
    wcnf_file = f"{input_file}.wcnf"
    subprocess.check_call(["cat", input_file])
    subprocess.check_call(["cat", wcnf_file])
    print(f"{wcnf_file} eixsts? {os.path.exists(wcnf_file)}")
    cmd_solver = [prog_solver, wcnf_file]
    # subprocess.check_call(cmd_solver)
    # subprocess.call(cmd_solver)
    time_beg = time.time()
    res = subprocess.run(cmd_solver, stdout=subprocess.PIPE).stdout
    total_time = time.time() - time_beg
    last_line = res.split(b"\n")[-2]
    ans = list(map(int, last_line[1:].split()))
    print(last_line)
    print(ans, num_pos(ans))
    date = str(datetime.datetime.now())
    status = "success"
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    # todo
    cur.execute("insert")
    # res = subprocess.check_output(cmd_solver)

    # print(res)
    # with tempfile.NamedTemporaryFile() as fp:
    #     # make file
    #     input_file = f'data/tmp/{x}.txt'
    #     fp.write(x)
    #     # make sat file
    #     args_wcnf = ["to_wcnf", "-i", fp.name]
    #     subprocess.check_call(args_wcnf)
    #     # run solver
    #     wcnf_file = f"{fp.name}.wcnf"
    #     subprocess.check_call(["cat", fp.name])
    #     subprocess.check_call(["cat", wcnf_file])
    #     print(f"{wcnf_file} eixsts? {os.path.exists(wcnf_file)}")
    #     args_solver = [prog_solver, wcnf_file]
    #     # subprocess.check_output(args_solver)
    #     # res = subprocess.check_output(args_solver)
    #     # print(res)

    # make sat file
    # run solver
    # insert result to db
    pass


def make_table():
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    # cur.execute(
    #     # "CREATE TABLE attr_bench (date text, status text, prog, text, text_len, time, attr_size, sol_nvars, sol_nhard, sol_nsoft)"
    #     "CREATE TABLE attr_bench (date text, status text)"
    # )
    # cur.execute("desc attr_bench")
    # cur.execute("show tables")
    # cur.execute("select name from sqlite_master where type = 'table';")
    for row in cur.fetchall():
        print(row)
    con.commit()
    con.close()


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
    # todo
    json = exp.to_dict()  # type: ignore
    n = len(json.values())
    # print(json)
    # print(f"({', '.join('?' for _ in range(n))})")
    # print(tuple(json.values()))
    # print(f"INSERT INTO attr_bench VALUES ({', '.join('?' for _ in range(n))})")
    cur.execute(
        f"INSERT INTO attr_bench VALUES ({', '.join('?' for _ in range(n))})",
        tuple(map(str, json.values())),
    )
    # print(cur.execute("select * from attr_bench").fetchone())
    con.commit()
    # con.close()


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


def wip():
    clear_table()
    benchmark_program(30, "solver", "data/calgary_pref/bib-50", "out/hoge.txt")

    con = sqlite3.connect(dbname)
    cur = con.cursor()

    import pandas as pd

    df = pd.read_sql_query("SELECT * FROM attr_bench", con)
    df.to_csv("out/hoge.csv", index=False)


def clear_table():
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    exp = AttractorExp.create()
    d = exp.to_dict()  # type: ignore

    cur.execute("DROP TABLE attr_bench")
    cur.execute(f"CREATE TABLE attr_bench ({', '.join(key for key in d.keys())})")


def export_csv(out_file):
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    import pandas as pd

    df = pd.read_sql_query("SELECT * FROM attr_bench", con)
    df.to_csv(out_file, index=False)


def main():
    # wip()
    # sys.exit(0)
    clear_table()
    args = parse_args()
    benchmark_mul(args.timeout, algos, files, args.output, args.n_jobs)
    export_csv(args.output)
    # run_solver("abba")
    # make_table()
    # for i in range(1, 5):
    #     for x in gen(i):
    #         print(x)


if __name__ == "__main__":
    main()
