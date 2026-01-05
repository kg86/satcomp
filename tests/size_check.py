"""Verify that the registered solvers output the correct size for a given file."""

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
        "bidirectional_var0",
        Measure.bms,
        "uv run src/bidirectional_solver_var0.py --file {filename}",
    ),
    Solver(
        "bidirectional_var1",
        Measure.bms,
        "uv run src/bidirectional_solver_var1.py --file {filename}",
    ),
    Solver(
        "bidirectional_var2",
        Measure.bms,
        "uv run src/bidirectional_solver_var2.py --file {filename}",
    ),
    # Solver(
    #     "bidirectional-fast",
    #     Measure.bms,
    #     "uv run src/bidirectional_fast.py --file {filename}",
    # ),
    Solver(
        "slp",
        Measure.slp,
        "uv run src/slp_solver.py --file {filename}",
    ),
    Solver(
        "slp-fast",
        Measure.slp,
        "uv run src/slp_fast.py --file {filename}",
    ),
]


def compute_size(cmd) -> int:
    return int(
        subprocess.check_output(cmd + " | jq '.factor_size'", shell=True)
        .strip()
        .decode("utf8")
    )


def make_tsv(files: List[str]):
    solvers = [
        Solver(
            "attractor",
            Measure.attractor,
            "uv run src/attractor_solver.py --file {filename} --algo min | jq '.factor_size'",
        ),
        Solver(
            "bidirectional_var0",
            Measure.bms,
            "uv run src/bidirectional_solver_var0.py --file {filename} | jq '.factor_size'",
        ),
        Solver(
            "slp",
            Measure.slp,
            "uv run src/slp_solver.py --file {filename} | jq '.factor_size'",
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
        true_size = int(true_size)
        for solver in SOLVERS:
            print(f"verify {filename} {measure} {true_size} {solver.name}")
            if solver.measure == measure:
                size = compute_size(solver.cmd.format(filename=filename))
                if true_size != size:
                    msg = f"the output size of {solver.name} for {filename} is expected {true_size}, but is actually {size}"
                    raise Exception(msg)
