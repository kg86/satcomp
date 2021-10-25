# Compute minimum string attractors by pysat

import argparse
import sys
import time
import pprint
import json

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

from mytimer import Timer


def min_substr_hist(min_substrs):
    """
    plot histogram of string lengths
    """
    # histogram
    th = 20
    nmin_substrs_th1 = [l for b, l in min_substrs if l < th]
    nmin_substrs_th2 = [l for b, l in min_substrs if l >= th]
    print(f"# of min_substrs whose length < {th} = {len(nmin_substrs_th1)}")
    print(f"# of min_substrs whose length >= {th} = {len(nmin_substrs_th2)}")
    # nmin_substrs = list(map(len, min_substrs))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(nmin_substrs_th1, bins=50)
    fig.savefig("./out/substrs.png")


def attractor_of_size(text: bytes, k: int, op: str):
    """
    verify if there is an attractor of size `k` for `text`.
    """
    assert op in ["exact", "atmost"]
    res = dict()
    n = len(text)
    timer = Timer()

    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    timer.record("min substrs")
    min_substr_hist(min_substrs)

    print(f"text length = {len(text)}")
    print(f"# of min substrs = {len(min_substrs)}")

    # run solver
    cnf = CNF()
    for b, l in min_substrs:
        lcp_range = stralgo.get_lcprange(lcp, isa[b], l)
        occs = [sa[i] for i in range(lcp_range[0], lcp_range[1] + 1)]
        cnf.append(list(set(occ + i + 1 for occ in occs for i in range(l))))
    timer.record("clauses")

    print(f"n of clauses={len(cnf.clauses)}, # of vars={cnf.nv}")
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
    print("clauses for atmost", len(exclauses.clauses))
    cnf.extend(exclauses.clauses)
    print(f"# of clauses={len(cnf.clauses)}, # of vars={cnf.nv}")
    solver = Solver()
    solver.append_formula(cnf.clauses)

    print("\nsolver runs")
    if solver.solve():
        sol = solver.get_model()
        assert sol is not None
        attractor = list(filter(lambda x: 0 < x <= n, sol))
        res["# of attractors"] = len(attractor)
        res["attractor"] = attractor
        # print(attractor)
        print(f"#attractor = {len(attractor)}")
    timer.record("solver run")
    res["text length"] = n
    res["# of string attractors"] = k
    res["# of min substrs"] = len(min_substrs)
    res["# of clauses"] = len(cnf.clauses)
    res["# of variables"] = cnf.nv
    res["times"] = timer.times
    return res


def min_attractor_WCNF(text: bytes, timer: Timer):
    res = dict()
    n = len(text)
    res["text length"] = n

    # run fast
    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    # timer.record("string indice")
    print(f"text length = {len(text)}")
    print(f"# of min substrs = {len(min_substrs)}")

    min_substr_hist(min_substrs)
    timer.record("min substrs")
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
    timer.record("clauses")
    return wcnf


def min_attractor(text: bytes):
    # res = dict()
    # n = len(text)
    # res["text length"] = n
    # timer = Timer()

    # # run fast
    # sa = stralgo.make_sa_MM(text)
    # isa = stralgo.make_isa(sa)
    # lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    # min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    # # timer.record("string indice")
    # print(f"text length = {len(text)}")
    # print(f"# of min substrs = {len(min_substrs)}")

    # min_substr_hist(min_substrs)
    # timer.record("min substrs")
    # res["# of min substrs"] = len(min_substrs)

    # # run solver
    # wcnf = WCNF()
    # for b, l in min_substrs:
    #     lcp_range = stralgo.get_lcprange(lcp, isa[b], l)
    #     occs = [sa[i] for i in range(lcp_range[0], lcp_range[1] + 1)]
    #     # hard clause
    #     wcnf.append(list(set(occ + i + 1 for occ in occs for i in range(l))))
    # for i in range(n):
    #     # soft clause
    #     wcnf.append([-(i + 1)], weight=1)
    # timer.record("clauses")
    # rc2 = RC2(wcnf, solver="g3")
    res = dict()
    timer = Timer()
    wcnf = min_attractor_WCNF(text, timer)
    rc2 = RC2(wcnf)
    sol = rc2.compute()
    assert sol is not None
    timer.record("solver run")

    attractor = list(filter(lambda x: x > 0, sol))
    print(f"the size of minimum attractor = {len(attractor)}")
    print("minimum attractor is", attractor)
    res["times"] = timer.times
    res["# of minimum attractors"] = len(attractor)
    res["minimum attractor"] = attractor
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum String Attractors.")
    parser.add_argument("--file", type=str, help="input file", default="")
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
        args.file == ""
        or args.algo not in ["exact", "atmost", "min"]
        or (args.algo in ["exact", "atmost"] and args.size <= 0)
    ):
        parser.print_help()
        sys.exit()

    return args


if __name__ == "__main__":
    args = parse_args()

    text = open(args.file, "rb").read()
    if args.algo in ["exact", "atmost"]:
        res = attractor_of_size(text, args.size, args.algo)
    elif args.algo == "min":
        res = min_attractor(text)
    else:
        assert False
    res["file name"] = args.file

    if args.output == "":
        res["minimum attractor"] = "omitted"
        pp.pprint(res)
    else:
        with open(args.output, "w") as f:
            json.dump(res, f, ensure_ascii=False)
