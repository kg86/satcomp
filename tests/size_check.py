"""Verify that the registered solvers output the correct size for a given file."""

import json
import subprocess
import sys
from typing import List
import csv
import enum
import dataclasses


class Measure(str, enum.Enum):
    attractor = "attractor"
    bms = "bms"
    slp = "slp"


@dataclasses.dataclass
class Solver:
    name: str
    measure: Measure
    cmd: str


SOLVERS = [
    Solver(
        "attractor",
        Measure.attractor,
        "uv run src/attractor_solver.py --file {filename} --algo min",
    ),
    Solver(
        "bidirectional-fast",
        Measure.bms,
        "uv run src/bidirectional_fast.py --file {filename}",
    ),
    Solver(
        "slp-fast",
        Measure.slp,
        "uv run src/slp_fast.py --file {filename}",
    ),
]


def compute_size(cmd) -> int:
    output = subprocess.check_output(cmd, shell=True).strip().decode("utf8")
    parsed = json.loads(output)
    return int(parsed["factor_size"])


def make_tsv(files: List[str]):
    solvers = [
        Solver(
            "attractor",
            Measure.attractor,
            "uv run src/attractor_solver.py --file {filename} --algo min",
        ),
        Solver(
            "bidirectional-fast",
            Measure.bms,
            "uv run src/bidirectional_fast.py --file {filename}",
        ),
        Solver(
            "slp-fast",
            Measure.slp,
            "uv run src/slp_fast.py --file {filename}",
        ),
    ]
    writer = csv.writer(sys.stdout, delimiter="\t")
    writer.writerow(["filename", "measure", "size"])
    for file in files:
        for solver in solvers:
            size = compute_size(solver.cmd.format(filename=file))
            writer.writerow([file, solver.measure, size])
            sys.stdout.flush()


if __name__ == "__main__":
    prog = sys.argv[1]
    if prog == "make_tsv":
        filenames = sys.argv[2:]
        make_tsv(filenames)
    elif prog == "verify":
        filename, measure, true_size = sys.argv[2:]
        if measure not in Measure.__members__:
            raise Exception(f"Invalid measure: {measure}")
        measure = Measure[measure]

        true_size = int(true_size)
        target_solvers = [solver for solver in SOLVERS if solver.measure == measure]
        for solver in target_solvers:
            print(f"verify {filename} {measure} {true_size} {solver.name}")
            size = compute_size(solver.cmd.format(filename=filename))
            if true_size != size:
                raise Exception(
                    f"the output size of {solver.name} for {filename} is expected {true_size}, but is actually {size}"
                )
