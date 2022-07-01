import csv
import sqlite3

import lz_bench
import attractor_bench
import bidirectional_bench

dbname = "out/satcomp.db"


def comp_bench(out_file: str, target_key: str, target_none: str):
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    res = dict()
    cur.execute(f"select file_name from {lz_bench.dbtable}")
    header = ["file", "file_len"] + lz_bench.algos + ["attractor", "bidirectional"]
    files = sorted(
        list(
            set(
                x[0]
                for x in cur.execute(
                    f"select file_name from {lz_bench.dbtable}"
                ).fetchall()
            )
        )
    )
    lines = []

    def get_values(keys, table, file):
        query = f"select {','.join(keys)} from {table} WHERE file_name = '{file}'"
        res = cur.execute(query).fetchone()
        if res == None:
            return ["None" for _ in keys]
        return res

    for file in files:
        line = dict()
        line["file"] = file
        # lz
        for algo in lz_bench.algos:
            status, file_len, factor_size = cur.execute(
                f"select status, file_len, {target_key} from {lz_bench.dbtable} WHERE file_name = '{file}' and algo='{algo}'"
            ).fetchone()
            line["file_len"] = file_len
            if factor_size == target_none:
                line[algo] = status
            else:
                line[algo] = factor_size

        # attractor
        status, target = get_values(
            ["status", target_key], attractor_bench.dbtable, file
        )
        line["attractor"] = status if target == target_none else target

        # bidirectional
        status, target = get_values(
            ["status", target_key], bidirectional_bench.dbtable, file
        )
        line["bidirectional"] = status if target == target_none else target

        lines.append([line[key] for key in header])

    with open(out_file, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(lines)


if __name__ == "__main__":
    comp_bench("out/benchmark_time.csv", "time_total", "0")
    comp_bench("out/benchmark_size.csv", "factor_size", "0")
