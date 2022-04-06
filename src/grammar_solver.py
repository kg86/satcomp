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

from repair import repair

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
    leaf = auto()  # true iff (i,j) is a leaf of POSLP
    internal = auto()  # true iff (i,j) is an internal node of POSLP


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
            SLPLiteral.internal: self.verify_internal,
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

    def verify_leaf(self, obj: Tuple[str, int, int]):
        # obj = (name, beg, end)
        assert len(obj) == 3
        assert obj[0] == self.lits.leaf
        assert 0 <= obj[1] < obj[2] <= self.n

    def verify_internal(self, obj: Tuple[str, int, int]):
        # obj = (name, beg, end)
        assert len(obj) == 3
        assert obj[0] == self.lits.internal
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



    lm = SLPLiteralManager(text)
    wcnf.append([lm.getid(lm.lits.true)])
    wcnf.append([-lm.getid(lm.lits.false)])

    # register all literals (except auxiliary literals) to literal manager
    lits = []
    for i in range(0, n):
        for j in range(i + 1, n + 1):
            lits.append(lm.newid(lm.lits.node, i, j))
            lits.append(lm.newid(lm.lits.leaf, i, j))
            lits.append(lm.newid(lm.lits.internal, i, j))

    #############################################################################
    # hard clauses ##############################################################
    #############################################################################

    # whole string is always the root node of POSLP
    wcnf.append([lm.getid(lm.lits.node, 0, n)])
    if n > 1:
        wcnf.append([-lm.getid(lm.lits.leaf, 0, n)])
        wcnf.append([lm.getid(lm.lits.internal, 0, n)])
    else:
        wcnf.append([lm.getid(lm.lits.leaf, 0, n)])
        wcnf.append([-lm.getid(lm.lits.internal, 0, n)])

    # wcnf.append([lm.getid(lm.lits.leaf, 0, 3)])
    # node(i, j) : true iff [i,j] is node of SOSLP
    # leaf(i, j) : true iff [i,j] is leaf of SOSLP
    # internal(i, j) : true iff [i,j] is internal node of SOSLP
    # 1. if (leaf(i, j) = true or internal(i,j) = true) then node(i, j) = true
    # 2. if leaf(i, j) = true then node(i',j') = false for all proper subintervals of [i,j]
    # 3. if j - i = 1 then internal(i,j) = false
    # 4. if j - i > 1 and internal(i, j) = true then for exactly one k, (node(i,k) and node(k,j)) is true
    # 5. for each distinct substring, if leaf(i,j)  then there must exists internal(i',j') (i' < i) with same substring
    for i in range(0, n):
        for j in range(i + 1, n + 1):
            # print(f"i={i}, j={j}")
            assert 0 <= i < j <= n
            node_ij = lm.getid(lm.lits.node, i, j)
            leaf_ij = lm.getid(lm.lits.leaf, i, j)
            internal_ij = lm.getid(lm.lits.internal, i, j)
            # 1. if (leaf(i, j) = true or internal(i,j)) then node(i, j) = true
            wcnf.append(pysat_if(leaf_ij, node_ij))
            wcnf.append(pysat_if(internal_ij, node_ij))

            # both leaf_ij and internal_ij cannot be true
            wcnf.append([-leaf_ij, -internal_ij])
            # if node_ij = true then leaf_ij or internal_ij must be true
            wcnf.append([-node_ij, leaf_ij, internal_ij])

            # print(has(wcnf))

            # 2. if leaf(i, j) = true then node(i',j') = false for all proper subintervals of [i,j]
            subintervals = []
            for iprime in range(i, j):
                for jprime in range(iprime + 1, j + 1):
                    if iprime == i and jprime == j:
                        continue
                    subintervals.append(lm.getid(lm.lits.node, iprime, jprime))
            # for k in range(i + 1, j):
            #    subintervals.append(lm.getid(lm.lits.node, i, k))
            #    subintervals.append(lm.getid(lm.lits.node, k, j))
            or_subintervals, clauses = pysat_or(lm.newid, subintervals)
            wcnf.extend(clauses)
            wcnf.append(pysat_if(leaf_ij, -or_subintervals))
            # print(has(wcnf))

            if j - i == 1:
                # 3. if j - i = 1 then internal(i,j) = false
                wcnf.append([-internal_ij])
                # wcnf.append(pysat_if(node_ij, leaf_ij))
            else:
                # 4. if j - i > 1 and internal(i, j) = true then for exactly one k, (node(i,k) and node(k,j)) is true
                children_candidates = []
                for k in range(i + 1, j):
                    candidate, clauses = pysat_and(
                        lm.newid,
                        [lm.getid(lm.lits.node, i, k), lm.getid(lm.lits.node, k, j)],
                    )
                    wcnf.extend(clauses)
                    children_candidates.append(candidate)
                has_children, clauses = pysat_exactlyone(lm, children_candidates)
                wcnf.extend(clauses)
                wcnf.append(pysat_if(internal_ij, has_children))

    # (assume j - i > 1)
    # 5. for each distinct substring, if leaf(i,j) then there must exists internal(i',j') (i' < i) with same substring
    for substr in set(stralgo.substr(text)):
        if len(substr) < 2:
            continue
        occs = stralgo.occ_pos_naive(text, substr)
        occs.sort()
        m = len(substr)
        # the first occurrence cannot be a leaf
        wcnf.append([-lm.getid(lm.lits.leaf, occs[0], occs[0] + m)])
        # if a non-first occurrence is a leaf, then there must exist a previous occurrence that is internal
        for occindex in range(1, len(occs)):
            leaf_ij = lm.getid(lm.lits.leaf, occs[occindex], occs[occindex] + m)
            prevoccs = occs[:occindex]
            internals_of_prev = [lm.getid(lm.lits.internal, i, i + m) for i in prevoccs]
            exists_prev_internal, clauses = pysat_name_cnf(
                lm, [pysat_atleast_one(internals_of_prev)]
            )
            wcnf.extend(clauses)
            wcnf.append(pysat_if(leaf_ij, exists_prev_internal))

    # soft clause: minimize number of internal nodes
    for i in range(n):
        for j in range(i + 1, n + 1):
            internal_ij = lm.getid(lm.lits.internal, i, j)
            wcnf.append([-internal_ij], weight=1)

    internals=[]
    for i in range(n):
        for j in range(i + 1, n + 1):
            internals.append(lm.getid(lm.lits.internal, i, j))

    repair_size = repair(text)
    xm, _ = pysat_atmost(lm, internals, repair_size)
    wcnf.append([xm])

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
            internal_ij = lm.getid(lm.lits.internal, i, j)
            if node_ij in solset:
                if leaf_ij not in solset:
                    internal_nodes.append(node_ij)
                print(
                    lm.id2str(node_ij),
                    f"isleaf={leaf_ij in solset}",
                    f"isinternal={internal_ij in solset}",
                )
                result.append(lm.id2obj(node_ij))
    print(result)
    if exp:
        exp.time_total = time.time() - total_start
        exp.time_prep = time_prep
        exp.factors = result
        exp.factor_size = len(internal_nodes) + len(set(text))
        exp.fill(wcnf)
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
