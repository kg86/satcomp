from mysat import *
import stralgo
import argparse
import os
import sys
import json
import time
from enum import auto

from attractor import AttractorType
from attractor_bench_format import AttractorExp

from typing import Optional, Dict, List
from logging import CRITICAL, getLogger, DEBUG, INFO, StreamHandler, Formatter

from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType, ITotalizer
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import matplotlib

# prevend appearing gui window
matplotlib.use("Agg")


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
FORMAT = "[%(lineno)s - %(funcName)10s() ] %(message)s"
formatter = Formatter(FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


class SLPLiteral(Enum):
    true = Literal.true
    false = Literal.false
    auxlit = Literal.auxlit
    phrase = auto()  # (i,l) (representing T[i:i+l)) is phrase of grammar parsing
    pstart = auto()  # i is a starting position of a phrase of grammar parsing
    ref = auto()  # (i,j,l): phrase (i,i+l) references T[j,j+l)  (T[j,j+l] <- T[i,i+l])
    referred = auto()  # (i,l): T[i,i+l) is referenced by some phrase


class SLPLiteralManager(LiteralManager):
    """
    Manage literals used for solvers.
    """

    def __init__(self, text: bytes):
        self.text = text
        self.n = len(self.text)
        self.lits = SLPLiteral
        super().__init__(self.lits)

    def newid(self, *obj) -> int:
        res = super().newid(*obj)
        return res


def compute_lpf(text: bytes):  # non-self-referencing lpf
    n = len(text)
    lpf = []
    for i in range(0, n):
        lpf.append(0)
        for j in range(0, i):
            for l in range(1, i - j + 1):
                if text[i : i + l] == text[j : j + l] and l > lpf[i]:
                    lpf[i] = l
    # print(f"{lpf}")
    return lpf


def smallest_SLP_WCNF(text: bytes) -> WCNF:
    """
    Compute the max sat formula for computing the smallest SLP
    """
    n = len(text)
    logger.info(f"text length = {len(text)}")
    wcnf = WCNF()

    lm = SLPLiteralManager(text)
    wcnf.append([lm.getid(lm.lits.true)])
    wcnf.append([-lm.getid(lm.lits.false)])

    lpf = compute_lpf(text)
    # defining the literals  ########################################
    # ref(i,j,l): defined for all i,j,l s.t. T[i:i+l) = T[j:j+l)
    refs = {}
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            for l in range(2, lpf[j] + 1):
                if text[i : i + l] == text[j : j + l]:
                    # print(f"1. ref: {i} <- {j} len = {l}")
                    lm.newid(lm.lits.ref, j, i, l)
                    if not (i, l) in refs:
                        refs[i, l] = []
                    refs[i, l].append(j)
    # pstart(i)
    # phrase(i,l)
    phrases = []
    for i in range(0, n + 1):
        lm.newid(lm.lits.pstart, i)  # start
        if i < n:
            # for l in range(1, lpf[i] + 1):
            for l in range(2, n - i + 1):
                # print(f"1. phrase: {i}, {l}")
                phrases.append((i, l))
                lm.newid(lm.lits.phrase, i, l)
                if l > lpf[i]:  # a first occurrence cannot be a phrase
                    # print(f"1. phrase: {i}, {l} : always false")
                    wcnf.append([-lm.getid(lm.lits.phrase, i, l)])

    for i in range(0, n):
        if lpf[i] == 0:
            wcnf.append([lm.getid(lm.lits.pstart, i)])
    wcnf.append([lm.getid(lm.lits.pstart, n)])

    # // for 1 <= l <= lpf(i):
    # phrase(i,l) = true <=> pstart[i] = pstart[i+l] = true, pstart[i+1..i+l) = false
    for (i, l) in phrases:
        plst = [lm.getid(lm.lits.pstart, i), lm.getid(lm.lits.pstart, i + l)]
        plst2 = [i, i + l]
        for j in range(1, l):
            plst.append(-lm.getid(lm.lits.pstart, i + j))
            plst2.append(-(i + j))
        range_iff_startp, clauses = pysat_and(lm.newid, plst)
        wcnf.extend(clauses)
        # print(f"2. phrase: {i},{l}, plst={plst2}, range_iff_startp={range_iff_startp}")
        wcnf.extend(pysat_iff(lm.getid(lm.lits.phrase, i, l), range_iff_startp))

    # if phrase(i,l) = true there must be exactly one of ref(i,j,l) with i < j that is true
    for (i, l) in refs.keys():
        unique_source, clauses = pysat_exactlyone(
            lm, [lm.getid(lm.lits.ref, x, i, l) for x in refs[i, l]]
        )
        wcnf.extend(clauses)
        wcnf.append(pysat_if(lm.getid(lm.lits.phrase, i, l), unique_source))

    # if referred(i,l) = true, then for all other referred(k,l) with k < j and T[i:i+l) = T[k:k+l) must be false
    # not implemented as this is not a requirement

    # referred(i,l) = true if there is some j > i such that ref(j,i,l) = true
    referred = []
    for (i, l) in refs.keys():
        ref_sources, clauses = pysat_name_cnf(lm, [refs[i, l]])
        wcnf.extend(clauses)
        wcnf.append([-ref_sources, lm.newid(lm.lits.referred, i, l)])
        referred.append((i, l))

    # # if (occ,l) is a referred interval, it cannot be a phrase
    # # phrase(occ,l) is only defined if l <= lpf[occ]
    for (occ, l) in referred:
        if l <= lpf[occ]:
            # print(f"2: occ={occ}, l={l}")
            wcnf.append(
                [
                    -lm.getid(lm.lits.referred, occ, l),  # hoge
                    -lm.getid(lm.lits.phrase, occ, l),
                ]
            )

    # crossing intervals cannot be referred to at the same time.
    for (occ1, l1) in referred:
        for (occ2, l2) in referred:
            if occ1 < occ2 and occ2 < occ1 + l1 and occ1 + l1 < occ2 + l2:
                id1 = lm.getid(lm.lits.referred, occ1, l1)
                id2 = lm.getid(lm.lits.referred, occ2, l2)
                # print(f"({occ1},{l1}) XX ({occ2},{l2})")
                wcnf.append([-id1, -id2])

    # soft clauses: minimize of phrases
    for i in range(0, n):
        wcnf.append([-lm.getid(lm.lits.pstart, i)], weight=1)
    return lm, wcnf, phrases


def smallest_SLP(text: bytes, exp: Optional[AttractorExp] = None):
    """
    Compute the smallest SLP.
    """
    total_start = time.time()
    lm, wcnf, phrases = smallest_SLP_WCNF(text)
    print(f"WCNF constructed")
    rc2 = RC2(wcnf)
    time_prep = time.time() - total_start
    sol = rc2.compute()
    assert sol is not None

    result = []
    n = len(text)
    solset = set(sol)
    # print(f"sol={sol}")
    # print(f"solset={solset}")
    # print(result)

    posl = []
    for i in range(0, n + 1):
        x = lm.getid(lm.lits.pstart, i)
        if x in solset:
            # posl.append(f"{i}")
            posl.append(i)
        # elif -x in solset:
        # posl.append(f"-{i}")
    print(f"phrase start positions: {posl}")
    print(
        f"phrase start positions id: {[lm.getid(lm.lits.pstart, i) for i in range(0, n+1)]}"
    )

    phrasel = []
    print(f"*phrases={phrases}")
    for (occ, l) in phrases:
        x = lm.getid(lm.lits.phrase, occ, l)
        if x in solset:
            phrasel.append((occ, l))
    print(f"phrases: {phrasel}")

    slpsize = len(posl) - 2 + len(set(text))
    print(f"smallest grammar size = {slpsize}")

    if exp:
        exp.time_total = time.time() - total_start
        exp.time_prep = time_prep
        exp.sol_nvars = wcnf.nv
        exp.sol_nhard = len(wcnf.hard)
        exp.sol_nsoft = len(wcnf.soft)
        exp.factors = result
        exp.factor_size = 0  # len(internal_nodes) + len(set(text))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum SLP.")
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
        "--log_level",
        type=str,
        help="log level, DEBUG/INFO/CRITICAL",
        default="CRITICAL",
    )
    args = parser.parse_args()
    if (
        args.file == ""
        and args.str == ""
        # or args.algo not in ["exact", "atmost", "min"]
        # or (args.algo in ["exact", "atmost"] and args.size <= 0)
        # or (args.log_level not in ["DEBUG", "INFO", "CRITICAL"])
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

    slp = smallest_SLP(text, exp)

    # exp.factors = slp
    # exp.factor_size = len(slp)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
