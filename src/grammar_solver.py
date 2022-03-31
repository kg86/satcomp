import argparse
from dis import _HaveCodeOrStringType
import os
import sys
import json
import time
from enum import auto

from attractor import AttractorType
from attractor_bench_format import AttractorExp

from typing import Optional, Dict
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
from mysat import *


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
    node = auto()  # (i,j) is/not node (including leaf) of POSLP
    leaf = auto()  # (i,j) is/not leaf of POSLP (if true, node (i,j) must be true)


class SLPLiteralManager(LiteralManager):
    """
    Manage literals used for solvers.
    """

    def __init__(self, text: bytes, max_depth: int):
        self.text = text
        self.max_depth = max_depth
        self.n = len(self.text)
        self.lits = SLPLiteral
        self.verifyf = {
            SLPLiteral.node: self.verify_node,
            SLPLiteral.leaf: self.verify_leaf,
        }
        super().__init__(self.lits)

    def newid(self, *obj) -> int:
        res = super().newid(*obj)
        if len(obj) > 0 and obj[0] in self.verifyf:
            self.verifyf[obj[0]](obj)
        return res

    def verify_node(self, obj: Tuple[str, int, int]):
        # obj = (name, beg, end)
        assert len(obj) == 3
        assert obj[0] == self.lits.node
        assert 0 <= obj[1] < obj[2] <= self.n

    def verify_leaf(self, obj: Tuple[str, int]):
        # obj = (name, beg, end)
        assert len(obj) == 3
        assert obj[0] == self.lits.leaf
        assert 0 <= obj[1] < obj[2] < self.n


def smallest_grammar_WCNF(text: bytes) -> WCNF:
    """
    Compute the max sat formula for computing the smallest grammar.
    """
    n = len(text)
    logger.info(f"text length = {len(text)}")

    # substrs = substr(text)
    #
    lm = SLPLiteralManager(text, max_depth)
    wcnf.append([lm.getid(lm.lits.true)])
    wcnf.append([-lm.getid(lm.lits.false)])

    # register all literals (except auxiliary literals) to literal manager
    # lits = [lm.sym2id(lm.true)]
    lits = []

    wcnf = WCNF()
    for i in range(0, n):
        for j in range(i + 1, n + 1):
            lits.append(lm.newid(lm.node, i, j))
            lits.append(lm.newid(lm.leaf, i, j))

    # hard clauses
    wcnf.append(
        [lm.getid(lm.node, 0, n)]
    )  # whole string is always the root node of POSLP
    if n > 1:
        wcnf.append(
            [lm.getid(lm.leaf, 0, n)]
        )  # whole string is always the root node of POSLP
    else:
        wcnf.append(
            [-lm.getid(lm.leaf, 0, n)]
        )  # whole string is always the root node of POSLP

    # each node is a leaf, or has children
    # if node(i,j) = true then there is at most one i <= k < j such that both node(i,k+1) and node(k+1,j) are true,
    # and for all other i <= k' < j, k \neq k', node(i,k'+1) and node(k'+1,j) are false
    # -----
    # if node(i,j) = true then case1 = true,
    # case1: there is at most one case2s is true
    # case2: both node(i,k) and node(k,j) are true for i < k <= j such that
    # and for all other i <= k' < j, k \neq k', node(i,k'+1) and node(k'+1,j) are false
    # if case2 = true then case3 = true
    # case3: case4 = false
    # case4: (or node(i, k'+1), node(k'+1, j) for i <= k' < j, k \neq k')
    for i in range(0, n):
        for j in range(i + 1, n + 1):
            nid = lm.getid(lm.node, i, j)
            # pysat_if(nid, case1)
            # case2 list

            #

            case2_list = []
            for k in range(i + 1, j):
                case2, clauses = pysat_and(
                    lm.newid, [lm.getid(lm.node, i, k), lm.getid(lm.node, k, j)]
                )
                wcnf.extend(clauses)
                case2_list.append(case2)
            case1, clauses = pysat_and(
                CardEnc.atmost(case2_list, bound=1, vpool=lm.vpool)
            )
            wcnf.extend(clauses)

            # case2_list.append([case2_var(k), lm.getid(lm.node,i,k),-lm.getid(lm.node,k,j)])
            # case2_list.append([case2_var(k),-lm.getid(lm.node,i,k),lm.getid(lm.node,k,j)])

            for k in range(i + 1, j):
                # def. case2

                for k2 in range(i + 1, j):
                    if k == k2:
                        continue
                pass
            refi_left = [lm.getid(lm.node, i, k) for k in range(i + 1, j)]
            refi_right = [lm.getid(lm.node, k, j) for k in range(i + 1, j)]
            wcnf.extend(CardEnc.atmost(refi_left, bound=1, vpool=lm.vpool))
            wcnf.extend(CardEnc.atmost(refi_right, bound=1, vpool=lm.vpool))

    # soft clauses
    wcnf.append([-(i + 1)], weight=1)
    return wcnf


def smallest_grammar(text: bytes, exp: Optional[AttractorExp] = None):
    """
    Compute the smallest grammar.
    """
    total_start = time.time()
    wcnf = smallest_grammar_WCNF(text)
    rc2 = RC2(wcnf)
    time_prep = time.time() - total_start
    sol = rc2.compute()
    assert sol is not None

    attractor = AttractorType(list(x - 1 for x in filter(lambda x: x > 0, sol)))
    logger.info(f"the size of smallest SLP = {len(attractor)}")
    logger.info(f"smallest SLP is {attractor}")
    if exp:
        exp.time_total = time.time() - total_start
        exp.time_prep = time_prep
        exp.sol_nvars = wcnf.nv
        exp.sol_nhard = len(wcnf.hard)
        exp.sol_nsoft = len(wcnf.soft)
        exp.factors = attractor
        exp.factor_size = len(attractor)
    return attractor


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

    slp = smallest_grammar(text, exp)

    exp.factors = slp
    exp.factor_size = len(slp)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
