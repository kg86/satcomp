# verify the output of algorithm
# python size_check.py "filename, measure, size"

import subprocess
import sys
import typing
from typing import List
import csv


measures = ["attractor", "bidirectional", "slp"]

algos = { "attractor" : ["attractor_solver"],
    "bidirectional" : ["bms_solver", "bms_fast", "bms_plus"],
    "slp" : ["slp_solver", "slp_fast"]
    }


def compute_sizes(filename, measure) -> typing.List[int]:
    assert measure in measures
    assert measure in algos
    cmds = [f"pipenv run python src/{algo}.py --file {filename} | jq '.output_size'" for algo in algos[measure]]
    res = [int(subprocess.check_output(cmd, shell=True).strip().decode("utf8")) for cmd in cmds]
    return res


def make_tsv(files: List[str]):
    writer = csv.writer(sys.stdout, delimiter="\t")
    writer.writerow(["filename", "measure", "size"])
    for file in files:
        for measure in measures:
            size = compute_sizes(file, measure)[0]
            writer.writerow([file, measure, size])


if __name__ == "__main__":
    prog = sys.argv[1]
    if prog == "make_tsv":
        filenames = sys.argv[2:]
        make_tsv(filenames)
    elif prog == "verify":
        filename, measure, true_size = sys.argv[2:]
        true_size = int(true_size)
        sizes = compute_sizes(filename, measure)
        for size in sizes:
          if true_size != size:
              msg = f"the output size of {measure} for {filename} is expected {true_size}, but is actually {size}"
              raise Exception(msg)
