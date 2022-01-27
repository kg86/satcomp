import datetime
import time
from typing import Iterable
from joblib import Parallel, delayed
import sqlite3
import tempfile
import subprocess
import os

dbname = "attractor_bench.db"


def gen(n: int) -> Iterable[str]:
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


def run_solver(x: str):
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


def main():
    run_solver("abba")
    # make_table()
    # for i in range(1, 5):
    #     for x in gen(i):
    #         print(x)


if __name__ == "__main__":
    main()
