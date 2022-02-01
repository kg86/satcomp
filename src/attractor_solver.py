# Compute minimum string attractors by pysat

import argparse
import os
import sys
import pprint
import json
import time
from attractor import AttractorType
from attractor_bench_format import AttractorExp

pp = pprint.PrettyPrinter(indent=2)

import stralgo

from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType, ITotalizer
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import matplotlib

# prevend appearing gui window
matplotlib.use("Agg")
from typing import Optional, Dict

from mytimer import Timer

from logging import getLogger, DEBUG, INFO, StreamHandler, Formatter

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
FORMAT = "[%(lineno)s - %(funcName)10s() ] %(message)s"
formatter = Formatter(FORMAT)
handler.setFormatter(formatter)
# logger.setLevel(DEBUG)
logger.setLevel(INFO)
logger.addHandler(handler)


def min_substr_hist(min_substrs, th):
    """
    plot histogram of string lengths less than the threshold
    """
    # histogram
    nmin_substrs_th1 = [l for b, l in min_substrs if l < th]
    nmin_substrs_th2 = [l for b, l in min_substrs if l >= th]
    logger.info(f"# of min_substrs whose length < {th} = {len(nmin_substrs_th1)}")
    logger.info(f"# of min_substrs whose length >= {th} = {len(nmin_substrs_th2)}")
    # nmin_substrs = list(map(len, min_substrs))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(nmin_substrs_th1, bins=50)
    fig.savefig("./out/substrs.png")


def attractor_of_size(
    text: bytes, k: int, op: str, exp: Optional[AttractorExp] = None
) -> AttractorType:
    """
    Compute string attractor of the specified size (1-indexed)

    `k`: attractor size
    `op`: `exact` or `atmost`.
        `exact` computes string attractor whose size is `k`.
        `atmost` computes string attractor whose size is at most `k`.
    `exp`: experiment information
    """
    assert op in ["exact", "atmost"]
    n = len(text)
    total_start = time.time()

    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)

    logger.info(f"text length = {len(text)}")
    logger.info(f"# of min substrs = {len(min_substrs)}")

    cnf = CNF()
    for b, l in min_substrs:
        lcp_range = stralgo.get_lcprange(lcp, isa[b], l)
        occs = [sa[i] for i in range(lcp_range[0], lcp_range[1] + 1)]
        cnf.append(list(set(occ + i + 1 for occ in occs for i in range(l))))

    logger.info(f"n of clauses={len(cnf.clauses)}, # of vars={cnf.nv}")
    exclauses = None
    # add conditions that # of solutions is exact/atmost `k`.
    if op == "atmost":
        #
        # atmost = CardEnc.atmost(lits, bound=k, top_id=n + 1, encoding=EncType.seqcounter)
        # atmost = CardEnc.equals(
        #     lits, bound=k, top_id=n + 1, encoding=EncType.sortnetwrk
        # )  # o
        exclauses = CardEnc.atmost(
            # lits, bound=k, top_id=n + 1, encoding=EncType.cardnetwrk
            list(range(1, n + 1)),
            bound=k,
            top_id=n + 1,
            encoding=EncType.cardnetwrk,
        )  # o
        # atmost = CardEnc.atmost(lits, bound=k, top_id=n + 1, encoding=EncType.bitwise) # x
    elif op == "exact":
        # atmost = CardEnc.equals(lits, bound=k, top_id=n + 1, encoding=EncType.ladder) # x
        exclauses = CardEnc.equals(
            list(range(1, n + 1)), bound=k, top_id=n + 1, encoding=EncType.totalizer
        )  # o
        # atmost = CardEnc.equals(lits, bound=k, top_id=n + 1, encoding=EncType.mtotalizer) # o
        # atmost = CardEnc.equals(lits, bound=k, top_id=n + 1, encoding=EncType.kmtotalizer) # o
        # atmost = CardEnc.equals(lits, bound=k, top_id=n + 1, encoding=EncType.native) # x
        # atmost = CardEnc.equals(lits, bound=k, top_id=n + 1)
        # atmost = CardEnc.equals(lits, bound=k, top_id=n + 1, encoding=EncType.kmtotalizer)
    assert exclauses is not None
    logger.info(f"clauses for atmost {len(exclauses.clauses)}")
    cnf.extend(exclauses.clauses)
    logger.info(f"# of clauses={len(cnf.clauses)}, # of vars={cnf.nv}")
    solver = Solver()
    solver.append_formula(cnf.clauses)

    logger.info("solver runs")
    attractor = []
    attractor = AttractorType([])
    if solver.solve():
        sol = solver.get_model()
        assert sol is not None
        attractor = AttractorType(
            list(x - 1 for x in filter(lambda x: 0 < x <= n, sol))
        )
        logger.info(f"#attractor = {len(attractor)}")
    # timer.record("solver run")
    if exp:
        exp.time_total = time.time() - total_start
        assert isinstance(cnf.nv, int)
        exp.sol_nvars = int(cnf.nv)
        exp.sol_nhard = len(cnf.clauses)
        exp.sol_nsoft = 0
        exp.factors = attractor
        exp.factor_size = len(attractor)
    # if exp:
    #     exp["# of attractors"] = len(attractor)
    #     # exp["attractor"] = attractor
    #     exp["text length"] = n
    #     exp["# of string attractors"] = k
    #     exp["# of min substrs"] = len(min_substrs)
    #     exp["# of clauses"] = len(cnf.clauses)
    #     exp["# of variables"] = cnf.nv
    #     exp["times"] = timer.times
    return attractor


def min_attractor_WCNF(text: bytes) -> WCNF:
    """
    Compute the max sat formula for computing minimum string attractor (1-indexed)
    """
    res = dict()
    n = len(text)
    res["text length"] = n

    # run fast
    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    logger.info(f"text length = {len(text)}")
    logger.info(f"# of min substrs = {len(min_substrs)}")

    min_substr_hist(min_substrs, 20)
    # timer.record("min substrs")
    res["# of min substrs"] = len(min_substrs)

    # run solver
    wcnf = WCNF()
    for b, l in min_substrs:
        lcp_range = stralgo.get_lcprange(lcp, isa[b], l)
        occs = [sa[i] for i in range(lcp_range[0], lcp_range[1] + 1)]
        # hard clause
        wcnf.append(list(set(occ + i + 1 for occ in occs for i in range(l))))
    for i in range(n):
        # soft clause
        wcnf.append([-(i + 1)], weight=1)
    # timer.record("clauses")
    return wcnf


def min_attractor(text: bytes, exp: Optional[AttractorExp] = None) -> AttractorType:
    """
    Compute minimum string attractor (1-indexed)
    """
    total_start = time.time()
    # timer = Timer()
    wcnf = min_attractor_WCNF(text)
    rc2 = RC2(wcnf)
    time_prep = time.time() - total_start
    sol = rc2.compute()
    assert sol is not None
    # timer.record("solver run")

    # attractor = AttractorType(list(filter(lambda x: x > 0, sol)))
    attractor = AttractorType(list(x - 1 for x in filter(lambda x: x > 0, sol)))
    logger.info(f"the size of minimum attractor = {len(attractor)}")
    logger.info(f"minimum attractor is {attractor}")
    if exp:
        exp.time_total = time.time() - total_start
        exp.time_prep = time_prep
        exp.sol_nvars = wcnf.nv
        exp.sol_nhard = len(wcnf.hard)
        exp.sol_nsoft = len(wcnf.soft)
        exp.factors = attractor
        exp.factor_size = len(attractor)
    # if exp:
    #     exp["times"] = timer.times
    #     exp["# of minimum attractors"] = len(attractor)
    #     exp["minimum attractor"] = attractor
    return attractor


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum String Attractors.")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument(
        "--size",
        type=int,
        help="exact size or upper bound of attractor size to search",
        default=0,
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="[min: find a minimum string attractor, exact/atmost: find a string attractor whose size is exact/atmost SIZE]",
    )
    args = parser.parse_args()
    if (
        (args.file == "" and args.str == "")
        or args.algo not in ["exact", "atmost", "min"]
        or (args.algo in ["exact", "atmost"] and args.size <= 0)
    ):
        parser.print_help()
        sys.exit()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.str != "":
        text = args.str
    else:
        text = open(args.file, "rb").read()

    exp = AttractorExp.create()
    exp.algo = "solver"
    exp.file_name = os.path.basename(args.file)
    exp.file_len = len(text)

    if args.algo in ["exact", "atmost"]:
        attractor = attractor_of_size(text, args.size, args.algo, exp)
    elif args.algo == "min":
        attractor = min_attractor(text, exp)
    else:
        assert False

    exp.factors = attractor
    exp.factor_size = len(attractor)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
