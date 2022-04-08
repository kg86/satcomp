# Compute minimum string attractors by pysat

import argparse
import os
import sys
import json
import time
from attractor import AttractorType
from attractor_bench_format import AttractorExp

from typing import List, Optional, Dict
from logging import CRITICAL, getLogger, DEBUG, INFO, StreamHandler, Formatter

from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType, ITotalizer
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import matplotlib

# prevend appearing gui window
matplotlib.use("Agg")
import stralgo


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
FORMAT = "[%(lineno)s - %(funcName)10s() ] %(message)s"
formatter = Formatter(FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


def min_substr_hist(min_substrs, th):
    """
    Plot histogram of string lengths less than the threshold.
    """
    # histogram
    nmin_substrs_th1 = [l for b, l in min_substrs if l < th]
    nmin_substrs_th2 = [l for b, l in min_substrs if l >= th]
    logger.info(f"# of min_substrs whose length < {th} = {len(nmin_substrs_th1)}")
    logger.info(f"# of min_substrs whose length >= {th} = {len(nmin_substrs_th2)}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(nmin_substrs_th1, bins=50)
    fig.savefig("./out/substrs.png")


def attractor_of_size(
    text: bytes, k: int, op: str, exp: Optional[AttractorExp] = None
) -> AttractorType:
    """
    Compute string attractor of the specified size.
    If such attractor does not exist, return an empty list.

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
    if exp:
        exp.time_total = time.time() - total_start
        assert isinstance(cnf.nv, int)
        exp.factors = attractor
        exp.factor_size = len(attractor)
        exp.fill(wcnf)
    return attractor


def min_attractor_WCNF(text: bytes) -> WCNF:
    """
    Compute the max sat formula for computing the minimum string attractor.
    """
    n = len(text)

    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    logger.info(f"text length = {len(text)}")
    logger.info(f"# of min substrs = {len(min_substrs)}")

    wcnf = WCNF()
    for b, l in min_substrs:
        lcp_range = stralgo.get_lcprange(lcp, isa[b], l)
        occs = [sa[i] for i in range(lcp_range[0], lcp_range[1] + 1)]
        # hard clauses
        wcnf.append(list(set(occ + i + 1 for occ in occs for i in range(l))))
    for i in range(n):
        # soft clauses
        wcnf.append([-(i + 1)], weight=1)
    return wcnf


def min_attractor(
    text: bytes, exp: Optional[AttractorExp] = None, contain_list: List[int] = []
) -> AttractorType:
    """
    Compute the minimum string attractor.
    """
    total_start = time.time()
    wcnf = min_attractor_WCNF(text)
    for i in contain_list:
        wcnf.append([i])
    rc2 = RC2(wcnf)
    time_prep = time.time() - total_start
    sol = rc2.compute()
    assert sol is not None

    attractor = AttractorType(list(x - 1 for x in filter(lambda x: x > 0, sol)))
    logger.info(f"the size of minimum attractor = {len(attractor)}")
    logger.info(f"minimum attractor is {attractor}")
    if exp:
        exp.time_total = time.time() - total_start
        exp.time_prep = time_prep
        exp.factors = attractor
        exp.factor_size = len(attractor)
        exp.fill(wcnf)
    return attractor


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum String Attractors.")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument(
        "--contains",
        nargs="+",
        type=int,
        help="list of text positions that must be included in the string attractor, starting with index 1",
        default=[],
    )
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
    parser.add_argument(
        "--log_level",
        type=str,
        help="log level, DEBUG/INFO/CRITICAL",
        default="CRITICAL",
    )
    args = parser.parse_args()
    if (
        (args.file == "" and args.str == "")
        or args.algo not in ["exact", "atmost", "min"]
        or (args.algo in ["exact", "atmost"] and args.size <= 0)
        or (args.log_level not in ["DEBUG", "INFO", "CRITICAL"])
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

    if args.log_level == "DEBUG":
        logger.setLevel(DEBUG)
    elif args.log_level == "INFO":
        logger.setLevel(INFO)
    elif args.log_level == "CRITICAL":
        logger.setLevel(CRITICAL)

    exp = AttractorExp.create()
    exp.algo = "solver"
    exp.file_name = os.path.basename(args.file)
    exp.file_len = len(text)

    if args.algo in ["exact", "atmost"]:
        attractor = attractor_of_size(text, args.size, args.algo, exp)
    elif args.algo == "min":
        attractor = min_attractor(text, exp, args.contains)
    else:
        assert False

    exp.factors = attractor
    exp.factor_size = len(attractor)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
