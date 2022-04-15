from mysat import *
import stralgo
import argparse
import os
import sys
import json
import time
import functools
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
    ref = auto()  # (j,i,l): phrase (j,j+l) references T[i,i+l)  (T[i,i+l] <- T[j,j+l])
    referred = auto()  # (i,l): T[i,i+l) is referenced by some phrase


class SLPLiteralManager(LiteralManager):
    """
    Manage literals used for solvers.
    """

    def __init__(self, text: bytes):
        self.text = text
        self.n = len(self.text)
        self.lits = SLPLiteral
        self.verifyf = {
            SLPLiteral.phrase: self.verify_phrase,
            SLPLiteral.pstart: self.verify_pstart,
            SLPLiteral.ref: self.verify_ref,
            SLPLiteral.referred: self.verify_referred,
        }
        super().__init__(self.lits)

    def newid(self, *obj) -> int:
        res = super().newid(*obj)
        if len(obj) > 0 and obj[0] in self.verifyf:
            self.verifyf[obj[0]](*obj)
        return res

    def verify_phrase(self, *obj):
        # print(f"verify_phrase, {obj}")
        # obj = (name, i, l) (representing T[i:i+l)) is phrase of grammar parsing
        assert len(obj) == 3
        name, i, l = obj
        assert name == self.lits.phrase
        assert 0 <= i < self.n
        assert 0 < l <= self.n
        assert i + l <= self.n

    def verify_pstart(self, *obj):
        # print(f"verify_pstart, {obj}")
        # obj = (name, i) i is a starting position of a phrase of grammar parsing
        assert len(obj) == 2
        name, i = obj
        assert name == self.lits.pstart
        assert 0 <= i <= self.n

    def verify_ref(self, *obj):
        # print(f"verify_ref, {obj}")
        # obj = (name, j, i, l) phrase (j,j+l) references T[i,i+l)  (T[i,i+l] <- T[j,j+l])
        assert len(obj) == 4
        name, j, i, l = obj
        assert name == self.lits.ref
        assert 0 <= i < self.n
        assert i < i + l <= j < j + l <= self.n
        assert 0 < l <= self.n

    def verify_referred(self, *obj):
        # print(f"verify_referred, {obj}")
        # obj = (name, i, l) T[i,i+l) is referenced by some phrase
        assert len(obj) == 3
        name, i, l = obj
        assert name == self.lits.referred
        assert 0 <= i < self.n
        assert 0 < l <= self.n
        assert i + l <= self.n


def compute_lpf(text: bytes):  # non-self-referencing lpf
    """
    lpf[i] = length of longest prefix of text[i:] that occurs in text[0:i]
    """
    n = len(text)
    lpf = []
    for i in range(0, n):
        lpf.append(0)
        for j in range(0, i):
            l = 0
            while j + l < i and i + l < n and text[i + l] == text[j + l]:
                l += 1
            if l > lpf[i]:
                lpf[i] = l
    # print(f"{lpf}")
    return lpf


def smallest_SLP_WCNF(text: bytes):
    """
    Compute the max sat formula for computing the smallest SLP
    """
    n = len(text)
    logger.info(f"text length = {len(text)}")
    wcnf = WCNF()

    lm = SLPLiteralManager(text)
    wcnf.append([lm.getid(lm.lits.true)])
    wcnf.append([-lm.getid(lm.lits.false)])

    print("sloooow algorithm for lpf... (should use linear time algorithm)")
    lpf = compute_lpf(text)
    print("done")
    # defining the literals  ########################################
    # ref(i,j,l): defined for all i,j,l>1 s.t. T[i:i+l) = T[j:j+l)
    # pstart(i)
    # phrase(i,l) defined for all i, l > 1
    print("building literals")
    phrases = []
    for i in range(n + 1):
        lm.newid(lm.lits.pstart, i)  # definition of p_i
    for i in range(n):
        for l in range(1, n - i + 1):
            phrases.append((i, l))
            lm.newid(lm.lits.phrase, i, l)  # definition of f_{i,l}

    refs_by_referred = {}
    refs_by_referrer = {}
    for i in range(n):
        for j in range(i + 1, n):
            for l in range(2, lpf[j] + 1):
                if i + l <= j and text[i : i + l] == text[j : j + l]:
                    lm.newid(lm.lits.ref, j, i, l)  # definition of ref_{i<-j,l}
                    if not (i, l) in refs_by_referred:
                        refs_by_referred[i, l] = []
                    refs_by_referred[i, l].append(j)
                    if not (j, l) in refs_by_referrer:
                        refs_by_referrer[j, l] = []
                    refs_by_referrer[j, l].append(i)
    for (i, l) in refs_by_referred.keys():
        lm.newid(lm.lits.referred, i, l)
    print("done")
    # print(f"refs_by_referred = {refs_by_referred}")
    # print(f"refs_by_referrer = {refs_by_referrer}")
    # phrase(i,l) = true <=> pstart[i] = pstart[i+l] = true, pstart[i+1..i+l) = false
    # print("compute 1")
    print("building constraints")
    # // start constraint (1) ###############################
    for (i, l) in phrases:
        plst = [-lm.getid(lm.lits.pstart, (i + j)) for j in range(1, l)] + [
            lm.getid(lm.lits.pstart, i),
            lm.getid(lm.lits.pstart, (i + l)),
        ]
        range_iff_startp, clauses = pysat_and(lm.newid, plst)
        wcnf.extend(clauses)
        wcnf.extend(pysat_iff(lm.getid(lm.lits.phrase, i, l), range_iff_startp))
    # // end constraint (1) ###############################

    # // start constraint (2) ###############################
    # if phrase(j,l) = true there must be exactly one i < j such that ref(j,i,l) is true
    for (j, l) in refs_by_referrer.keys():
        clauses = CardEnc.atmost(
            [lm.getid(lm.lits.ref, j, i, l) for i in refs_by_referrer[j, l]],
            bound=1,
            vpool=lm.vpool,
        )
        wcnf.extend(clauses)
        clause = pysat_atleast_one(
            [lm.getid(lm.lits.ref, j, i, l) for i in refs_by_referrer[j, l]]
        )
        var_atleast, clause_atleast = pysat_name_cnf(lm, [clause])
        wcnf.extend(clause_atleast)
        phrase = lm.getid(lm.lits.phrase, j, l)
        wcnf.append(pysat_if(phrase, var_atleast))
    # for (j, l) in refs_by_referrer.keys():
    #     assert l > 1
    #     for i in refs_by_referrer[j, l]:
    #         print(f"{i}<-{j},{j+l}")
    #     unique_source, clauses = pysat_exactlyone(
    #         lm, [lm.getid(lm.lits.ref, j, i, l) for i in refs_by_referrer[j, l]]
    #     )
    #     wcnf.extend(clauses)
    #     wcnf.append(pysat_if(lm.getid(lm.lits.phrase, j, l), unique_source))
    # # // end constraint (2) ###############################
    # // start constraint (3) ###############################
    for (j, l) in refs_by_referrer.keys():
        for i in refs_by_referrer[j, l]:
            wcnf.append(
                pysat_if(
                    lm.getid(lm.lits.ref, j, i, l),  # ref_{i<-j,l}
                    lm.getid(lm.lits.phrase, j, l),  # f_{j,l}
                )
            )
    # // end constraint (3) ###############################

    # print("compute 3")
    # referred(i,l) = true iff there is some j > i such that ref(j,i,l) = true
    # // start constraint (4) ###############################
    for (i, l) in refs_by_referred.keys():
        assert l > 1
        ref_sources, clauses = pysat_or(
            lm.newid,
            [lm.getid(lm.lits.ref, j, i, l) for j in refs_by_referred[i, l]],
        )
        wcnf.extend(clauses)
        referredid = lm.getid(lm.lits.referred, i, l)
        wcnf.extend(pysat_iff(ref_sources, referredid))
    # // end constraint (4) ###############################
    # # if (occ,l) is a referred interval, it cannot be a phrase, but pstart[occ] and pstart[occ+l] must be true
    # # phrase(occ,l) is only defined if l <= lpf[occ]
    # // start constraint (5) ###############################
    referred = list(refs_by_referred.keys())
    for (occ, l) in referred:
        if l > 1:
            qid = lm.getid(lm.lits.referred, occ, l)
            wcnf.append(pysat_if(qid, -lm.getid(lm.lits.phrase, occ, l)))
            wcnf.append(pysat_if(qid, lm.getid(lm.lits.pstart, occ)))
            wcnf.append(pysat_if(qid, lm.getid(lm.lits.pstart, occ + l)))
    # // end constraint (5) ###############################
    # print("done")
    # print("compute 5")

    # // start constraint (6) ###############################
    # crossing intervals cannot be referred to at the same time.
    referred_by_bp = [[] for _ in range(n)]
    for (occ, l) in referred:
        referred_by_bp[occ].append(l)
    for lst in referred_by_bp:
        lst.sort(reverse=True)
    # print(f"referred_by_bp={referred_by_bp}")
    for (occ1, l1) in referred:
        for occ2 in range(occ1 + 1, occ1 + l1):
            for l2 in referred_by_bp[occ2]:
                assert l1 > 1 and l2 > 1
                assert occ1 < occ2 and occ2 < occ1 + l1
                if occ1 + l1 >= occ2 + l2:
                    break
                id1 = lm.getid(lm.lits.referred, occ1, l1)
                id2 = lm.getid(lm.lits.referred, occ2, l2)
                wcnf.append([-id1, -id2])
    # // end constraint (6) ###############################
    # print("done")

    # // start constraint (8) ###############################
    # a first occurrence of length longer than 2 cannot be a phrase
    for i in range(0, n):
        for l in range(2, n - i + 1):
            if l > lpf[i]:
                # print(f"1. phrase: {i}, {i+l} : always false")
                wcnf.append([-lm.getid(lm.lits.phrase, i, l)])
    # // end constraint (8) ###############################

    # // start constraint (9) ###############################
    for i in range(0, n):
        if lpf[i] == 0:
            # wcnf.append([lm.getid(lm.lits.pstart, i)])
            wcnf.append([lm.getid(lm.lits.phrase, i, 1)])

    wcnf.append([lm.getid(lm.lits.pstart, 0)])
    wcnf.append([lm.getid(lm.lits.pstart, n)])
    # // end constraint (9) ###############################

    # soft clauses: minimize of phrases
    for i in range(0, n):
        wcnf.append([-lm.getid(lm.lits.pstart, i)], weight=1)
    print("done")
    return lm, wcnf, phrases, refs_by_referrer


def postorder_cmp(x, y):
    i1 = x[0]
    j1 = x[1]
    i2 = y[0]
    j2 = y[1]
    # print(f"compare: {x} vs {y}")
    if i1 == i2 and i2 == j2:
        return 0
    if j1 <= i2:
        return -1
    elif j2 <= i1:
        return 1
    elif i1 <= i2 and j2 <= j1:
        return 1
    elif i2 <= i1 and j1 <= j2:
        return -1
    else:
        assert False


# given a list of nodes that in postorder of subtree rooted at root,
# find the direct children of [root_i,root_j) and add it to slp
# slp[j,l,i] is a list of nodes that are direct children of [i,j)
def build_slp_aux(nodes, slp):
    root = nodes.pop()
    root_i = root[0]
    root_j = root[1]
    # print(f"root_i,root_j = {root_i},{root_j}")
    children = []
    while len(nodes) > 0 and nodes[-1][0] >= root_i:
        # print(f"nodes[-1] = {nodes[-1]}")
        c = build_slp_aux(nodes, slp)
        children.append(c)
    children.reverse()
    slp[root] = children
    ##########################################################
    return root


# turn multi-ary tree into binary tree
def binarize_slp(root, slp):
    children = slp[root]
    numc = len(children)
    assert numc == 0 or numc >= 2
    if numc == 2:
        slp[root] = [binarize_slp(children[0], slp), binarize_slp(children[1], slp)]
    elif numc > 0:
        leftc = children[0]
        for i in range(1, len(children)):
            n = (root[0], children[i][1], None) if i < len(children) - 1 else root
            slp[n] = [leftc, children[i]]
            leftc = n
        for c in children:
            binarize_slp(c, slp)
    return root


def slp2str(root, slp):
    # print(f"root={root}")
    res = []
    (i, j, ref) = root
    if j - i == 1:
        res.append(ref)
    else:
        children = slp[root]
        if ref == None:
            assert len(children) == 2
            res += slp2str(children[0], slp)
            res += slp2str(children[1], slp)
        else:
            assert len(children) == 0
            n = (ref, ref + j - i, None)
            res += slp2str(n, slp)
    return res


def recover_slp(text: bytes, pstartl, refs_by_referrer):
    n = len(text)
    alph = set(text)
    referred = set((refs_by_referrer[j, l], l) for (j, l) in refs_by_referrer.keys())
    leaves = [(j, j + l, refs_by_referrer[j, l]) for (j, l) in refs_by_referrer.keys()]
    for i in range(len(pstartl) - 1):
        if pstartl[i + 1] - pstartl[i] == 1:
            leaves.append((pstartl[i], pstartl[i + 1], text[pstartl[i]]))
    internal = [(occ, occ + l, None) for (occ, l) in referred]
    nodes = leaves + internal
    nodes.append((0, n, None))

    # print(f"leaves: {leaves}")
    # print(f"internal: {internal}")
    # print(f"nodes: {nodes}")

    nodes.sort(key=functools.cmp_to_key(postorder_cmp))
    # print(f"leaves: {leaves}")
    # print(f"internal: {internal}")
    # print(f"nodes: {nodes}")
    slp = {}
    root = build_slp_aux(nodes, slp)
    binarize_slp(root, slp)
    return (root, slp)


def smallest_SLP(text: bytes, exp: Optional[AttractorExp] = None):
    """
    Compute the smallest SLP.
    """
    total_start = time.time()
    lm, wcnf, phrases, refs_by_referrer = smallest_SLP_WCNF(text)
    print(f"WCNF constructed")
    rc2 = RC2(wcnf)
    time_prep = time.time() - total_start
    sol_ = rc2.compute()
    assert sol_ is not None
    sol = set(sol_)
    # assert sol is not None

    result = []
    n = len(text)

    posl = []
    for i in range(0, n + 1):
        x = lm.getid(lm.lits.pstart, i)
        if x in sol:
            posl.append(i)
    # print(f"phrase start positions: {posl}")
    # print(
    #     f"phrase start positions id: {[lm.getid(lm.lits.pstart, i) for i in range(0, n+1)]}"
    # )

    phrasel = []
    # print(f"*phrases={phrases}")
    for (occ, l) in phrases:
        x = lm.getid(lm.lits.phrase, occ, l)
        if x in sol:
            phrasel.append((occ, occ + l))
    # print(f"phrases: {phrasel}")

    refs = {}
    for (j, l) in refs_by_referrer.keys():
        for i in refs_by_referrer[j, l]:
            if lm.getid(lm.lits.ref, j, i, l) in sol:
                # print(f"ref {i} <- {j},{j+l}")
                refs[j, l] = i
    # print(f"refs = {refs}")
    root, slp = recover_slp(text, posl, refs)
    # print(f"slp = {slp}")

    slpsize = len(posl) - 2 + len(set(text))
    # print(f"smallest slp size = {slpsize}")

    if exp:
        exp.time_total = time.time() - total_start
        exp.time_prep = time_prep
        # exp.factors = result
        exp.factor_size = slpsize  # len(internal_nodes) + len(set(text))
        exp.fill(wcnf)

    check = bytes(slp2str(root, slp))
    # print(f"check = {check}")
    assert check == text

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
        text = bytes(args.str, "utf-8")
    else:
        text = open(args.file, "rb").read()
    if args.log_level == "DEBUG":
        logger.setLevel(DEBUG)
    elif args.log_level == "INFO":
        logger.setLevel(INFO)
    elif args.log_level == "CRITICAL":
        logger.setLevel(CRITICAL)

    exp = AttractorExp.create()
    exp.algo = "slp-sat"
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
