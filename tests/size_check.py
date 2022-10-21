# verify the output of algorithm
# python size_check.py "filename, algo, size"

import subprocess
import sys
from typing import List
import csv


algos = ["attractor", "bidirectional", "slp"]


def compute_size(filename, algo) -> int:
    assert algo in algos
    if algo == "attractor":
        cmd = f"pipenv run python src/attractor_solver.py --file {filename} --algo min | jq '.output_size'"
    elif algo == "bidirectional":
        cmd = f"pipenv run python src/bms_solver.py --file {filename} | jq '.output_size'"
    elif algo == "slp":
        cmd = (
            f"pipenv run python src/slp_solver.py --file {filename} | jq '.output_size'"
        )
    else:
        assert False

    res = int(subprocess.check_output(cmd, shell=True).strip().decode("utf8"))
    return res


def make_tsv(files: List[str]):
    writer = csv.writer(sys.stdout, delimiter="\t")
    writer.writerow(["filename", "algo", "size"])
    for file in files:
        for algo in algos:
            size = compute_size(file, algo)
            writer.writerow([file, algo, size])


if __name__ == "__main__":
    prog = sys.argv[1]
    if prog == "make_tsv":
        filenames = sys.argv[2:]
        make_tsv(filenames)
    elif prog == "verify":
        filename, algo, true_size = sys.argv[2:]
        true_size = int(true_size)
        size = compute_size(filename, algo)
        if true_size != size:
            msg = f"the output size of {algo} for {filename} is expected {true_size}, but is actually {size}"
            raise Exception(msg)
