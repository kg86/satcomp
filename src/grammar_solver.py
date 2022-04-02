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

    def __init__(self, text: bytes):
        self.text = text
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
        assert 0 <= obj[1] < obj[2] <= self.n


def has(xs) -> bool:
    for clause in xs.hard:
        if 51 in clause:
            return True
    return False


def smallest_grammar_WCNF(text: bytes) -> WCNF:
    """
    Compute the max sat formula for computing the smallest grammar.
    """
    n = len(text)
    logger.info(f"text length = {len(text)}")
    wcnf = WCNF()

    # substrs = substr(text)
    #
    lm = SLPLiteralManager(text)
    wcnf.append([lm.getid(lm.lits.true)])
    wcnf.append([-lm.getid(lm.lits.false)])

    # register all literals (except auxiliary literals) to literal manager
    # lits = [lm.sym2id(lm.true)]
    lits = []

    for i in range(0, n):
        for j in range(i + 1, n + 1):
            lits.append(lm.newid(lm.lits.node, i, j))
            lits.append(lm.newid(lm.lits.leaf, i, j))

    # hard clauses
    wcnf.append(
        [lm.getid(lm.lits.node, 0, n)]
    )  # whole string is always the root node of POSLP
    if n > 1:
        wcnf.append(
            [-lm.getid(lm.lits.leaf, 0, n)]
        )  # whole string is always the root node of POSLP
    else:
        wcnf.append(
            [lm.getid(lm.lits.leaf, 0, n)]
        )  # whole string is always the root node of POSLP

    # wcnf.append([lm.getid(lm.lits.leaf, 0, 3)])
    # node(i,j) : true iff [i,j] is node of SOSLP
    # leaf(i,j) : true iff [i,j] is leaf of SOSLP
    #
    # 1. if leaf(i, j) = true then node(i, j) = true
    # 2. if leaf(i, j) = true then node(i',j') = false for all proper subintervals of [i,j]
    # 3. if node(i, j) = true and j-i = 1 then leaf(i,j) = true
    # 4. if node(i, j) = true and j-i > 1 and leaf(i, j) = false then for exactly one k, (node(i,k) and node(k,j)) is true

    for i in range(0, n):
        for j in range(i + 1, n + 1):
            # print(f"i={i}, j={j}")
            assert 0 <= i < j <= n
            node_ij = lm.getid(lm.lits.node, i, j)
            leaf_ij = lm.getid(lm.lits.leaf, i, j)

            # 1. if leaf(i, j) = true then node(i, j) = true
            wcnf.append(pysat_if(leaf_ij, node_ij))
            # print(has(wcnf))

            # 2. if leaf(i, j) = true then node(i',j') = false for all proper subintervals of [i,j]
            subintervals = []
            for iprime in range(i, j):
                for jprime in range(iprime + 1, j + 1):
                    if iprime == i and jprime == j:
                        continue
                    subintervals.append(lm.getid(lm.lits.node, iprime, jprime))
            or_subintervals, clauses = pysat_or(lm.newid, subintervals)
            wcnf.extend(clauses)
            wcnf.append(pysat_if(leaf_ij, -or_subintervals))
            # print(has(wcnf))

            if j - i == 1:
                # 3. if node(i, j) = true and j-i = 1 then leaf(i,j) = true
                wcnf.append(pysat_if(node_ij, leaf_ij))
                pass
            else:
                # 4. if node(i, j) = true and j-i > 1 and leaf(i, j) = false then for exactly one k, (node(i,k) and node(k,j)) is true
                children_candidates = []
                for k in range(i + 1, j):
                    candidate, clauses = pysat_and(
                        lm.newid,
                        [lm.getid(lm.lits.node, i, k), lm.getid(lm.lits.node, k, j)],
                    )
                    wcnf.extend(clauses)
                    children_candidates.append(candidate)
                # print(has(wcnf))
                has_children, clauses = pysat_exactlyone(lm, children_candidates)
                wcnf.extend(clauses)
                # print(3, has(wcnf))
                wcnf.append(pysat_if(has_children, -leaf_ij))
                # print(4, has(wcnf))
                if_cond, clauses = pysat_and(lm.newid, [node_ij, -leaf_ij])
                wcnf.extend(clauses)
                # print(5, has(wcnf))
                wcnf.append(pysat_if(if_cond, has_children))
            # print(has(wcnf))

    # (assume j - i > 1)
    # for each distinct substring, if there is a leaf(i,j) with that substring, then there must exists
    # a node(i',j') such that leaf(i',j') = false
    for substr in set(stralgo.substr(text)):
        if len(substr) < 2:
            continue
        occs = stralgo.occ_pos_naive(text, substr)
        m = len(substr)
        same_isleaf = [lm.getid(lm.lits.leaf, i, i + m) for i in occs]
        same_isinternal = []
        for i in occs:
            nvar, clauses = pysat_and(
                lm.newid,
                [lm.getid(lm.lits.node, i, i + m), -lm.getid(lm.lits.leaf, i, i + m)],
            )
            wcnf.extend(clauses)
            same_isinternal.append(nvar)

        exists_leaf, clauses = pysat_or(lm.newid, same_isleaf)
        wcnf.extend(clauses)

        exists_internal_node, clauses = pysat_name_cnf(
            lm, [pysat_atleast_one(same_isinternal)]
        )
        wcnf.extend(clauses)
        wcnf.append(pysat_if(exists_leaf, exists_internal_node))

    for i in range(n):
        for j in range(i + 1, n + 1):
            # if j - i == 1:
            #     continue
            leaf_ij = lm.getid(lm.lits.leaf, i, j)
            wcnf.append([-leaf_ij], weight=1)

    return lm, wcnf


def smallest_grammar(text: bytes, exp: Optional[AttractorExp] = None):
    """
    Compute the smallest grammar.
    """
    total_start = time.time()
    lm, wcnf = smallest_grammar_WCNF(text)
    rc2 = RC2(wcnf)
    time_prep = time.time() - total_start
    sol = rc2.compute()
    assert sol is not None

    result = []
    n = len(text)
    solset = set(sol)
    internal_nodes = []
    for i in range(n):
        for j in range(i + 1, n + 1):
            node_ij = lm.getid(lm.lits.node, i, j)
            leaf_ij = lm.getid(lm.lits.leaf, i, j)
            if node_ij in sol:
                if leaf_ij not in sol:
                    internal_nodes.append(node_ij)
                print(lm.id2str(node_ij), f"isleaf={leaf_ij in sol}")
                result.append(lm.id2obj(node_ij))

    # for node_str in [
    #     lm.id2str(x)
    #     for x in filter(
    #         lambda x: x > 0 and lm.contains(x) and lm.id2obj(x)[0] == lm.lits.node, sol
    #     )
    # ]:
    #     print(node_str)
    # logger.info(f"the size of smallest SLP = {len(attractor)}")
    # logger.info(f"smallest SLP is {attractor}")
    def grammar_size(factors) -> int:
        res = len(set(text))
        for factor in factors:
            if factor[2] - factor[1] > 1:
                res += 1
        return res

    print(result)
    if exp:
        exp.time_total = time.time() - total_start
        exp.time_prep = time_prep
        exp.sol_nvars = wcnf.nv
        exp.sol_nhard = len(wcnf.hard)
        exp.sol_nsoft = len(wcnf.soft)
        exp.factors = result
        exp.factor_size = len(internal_nodes) + len(set(text))
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

    slp = smallest_grammar(text, exp)

    # exp.factors = slp
    # exp.factor_size = len(slp)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
