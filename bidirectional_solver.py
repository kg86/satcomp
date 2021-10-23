# compute the minimum size bidirectional scheme by using SAT solver
import sys
import argparse
import os
from logging import getLogger, DEBUG, INFO, StreamHandler, Formatter
import time
from dataclasses import dataclass

from pysat.card import CardEnc, EncType
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

from mysat import *
from bidirectional import BiDirType, decode, BiDirExp, bd_info
from mytimer import Timer
import lz77


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
FORMAT = "[%(lineno)s - %(funcName)10s() ] %(message)s"
formatter = Formatter(FORMAT)
handler.setFormatter(formatter)
# logger.setLevel(DEBUG)
logger.setLevel(INFO)
logger.addHandler(handler)


@dataclass
class Literal:
    """
    Literals for solver to compute bidirectional scheme
    """

    link_to: str = "link_to"
    root: str = "root"
    fbeg: str = "factor_begin"
    ref: str = "ref"
    depth_ref: str = "depth_ref"


class BiDirLiteralManager(LiteralManager):
    """
    Manage literals to be used for solvers
    """

    def __init__(self, text: bytes, max_depth: int):
        self.lit = Literal()
        self.text = text
        self.max_depth = max_depth
        self.n = len(self.text)
        self.verifyf = {
            self.lit.link_to: self.verify_link,
            self.lit.root: self.verify_root,
            self.lit.ref: self.verify_ref,
            self.lit.depth_ref: self.verify_depth_ref,
        }
        super().__init__()

    def new_id(self, *obj) -> int:
        res = super().new_id(*obj)
        if len(obj) > 0 and obj[0] in self.verifyf:
            self.verifyf[obj[0]](obj)
        return res

    def verify_link(self, obj):
        # obj = (name, depth, form, to)
        assert len(obj) == 4
        assert obj[0] == self.lit.link_to
        assert 0 <= obj[1] < self.max_depth - 1
        assert 0 <= obj[2], obj[3] < self.n
        assert obj[2] != obj[3]
        assert self.text[obj[2]] == self.text[obj[3]]

    def verify_root(self, obj):
        # obj = (name, pos)
        assert len(obj) == 2
        assert 0 <= obj[1] < self.n

    def verify_ref(self, obj):
        # obj = (name, pos, ref_pos)
        assert len(obj) == 3
        assert obj[1] != obj[2]
        assert 0 <= obj[1], obj[2] < self.n
        assert self.text[obj[1]] == self.text[obj[2]]

    def verify_depth_ref(self, obj):
        # obj = (name, depth, ref_pos)
        assert len(obj) == 3
        assert 0 <= obj[1] < self.max_depth
        assert 0 <= obj[2] < self.n


def pysat_equal(lm: BiDirLiteralManager, bound: int, lits: list[int]):
    return CardEnc.equals(lits, bound=bound, encoding=EncType.pairwise, vpool=lm.vpool)


def sol2lits2(lm: BiDirLiteralManager, sol: dict[int, bool], lit_name: str) -> list:
    res = []
    for id, obj in lm.vpool.id2obj.items():
        assert isinstance(id, int)
        if obj[0] == lit_name:
            res.append((obj, sol[id]))

    return res


def sol2refs(lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes):
    n = len(text)
    occ = make_occa1(text)
    refs = dict()
    for i in range(n):
        for j in occ[text[i]]:
            if i == j:
                continue
            if sol[lm.getid(lm.lit.ref, i, j)]:
                refs[i] = j
                break
    logger.debug(f"refs={refs}")
    return refs


def show_sol(lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes):
    n = len(text)
    occ = make_occa1(text)
    pinfo = defaultdict(list)

    def link_to(i: int) -> list[int]:
        res = []
        for depth in range(lm.max_depth - 1):
            for j in occ[text[i]]:
                if i == j:
                    continue
                if sol[lm.getid(lm.lit.link_to, depth, i, j)]:
                    res.append((lm.lit.link_to, depth, i, j))
        return res

    for i in range(n):
        pinfo[i].append(chr(text[i]))
        for j in occ_others(occ, text, i):
            key = (lm.lit.ref, i, j)
            if sol[lm.getid(*key)]:
                pinfo[i].append(str(key))
        key = (lm.lit.root, i)
        lid = lm.getid(*key)
        if sol[lid]:
            pinfo[i].append(str(key))
        fbeg_key = (lm.lit.fbeg, i)
        if sol[lm.getid(*fbeg_key)]:
            pinfo[i].append(str(fbeg_key))
        for key in link_to(i):
            pinfo[i].append(f"{key}")
    for i in range(n):
        logger.debug(f"i={i} " + ", ".join(pinfo[i]))


def sol2bidirectional(
    lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes
) -> BiDirType:
    res = BiDirType([])
    fbegs = []
    n = len(text)
    refs = sol2refs(lm, sol, text)
    for i in range(n):
        if sol[lm.getid(lm.lit.fbeg, i)]:
            fbegs.append(i)
    fbegs.append(n)

    logger.debug(f"fbegs={fbegs}")
    for i in range(len(fbegs) - 1):
        flen = fbegs[i + 1] - fbegs[i]
        if flen == 1:
            res.append((-1, text[fbegs[i]]))
        else:
            res.append((refs[fbegs[i]], flen))
    return res


def make_occa1(text: bytes) -> dict[int, list[int]]:
    occ = defaultdict(list)
    for i in range(len(text)):
        occ[text[i]].append(i)
    return occ


def make_occa2(text: bytes) -> dict[bytes, list[int]]:
    match2 = defaultdict(list)
    for i in range(len(text) - 1):
        match2[text[i : i + 2]].append(i)
    return match2


def occ_others(occ1: dict[int, list[int]], text: bytes, i: int):
    """
    returns occurrences of text[i] except i.
    """
    for j in occ1[text[i]]:
        if i != j:
            yield j


def bidirectional_WCNF(text: bytes) -> tuple[BiDirLiteralManager, WCNF]:
    """
    returns weighted CNF to formulate the minimum size bidirectional scheme.
    """
    n = len(text)
    lz77fs = lz77.encode(text)
    logger.info("bidirectional_solver start")
    logger.info(f"# of text = {n}, # of lz77 = {len(lz77fs)}")

    occ1 = make_occa1(text)
    occ2 = make_occa2(text)

    max_depth = max(len(v) for v in occ1.values())
    lm = BiDirLiteralManager(text, max_depth)
    wcnf = WCNF()
    wcnf.append([lm.sym2id(lm.true)])
    wcnf.append([lm.sym2id(lm.false)])

    # register all literals (except auxiliary literals) to literal manager.
    lits = [lm.sym2id(lm.true)]
    for depth in range(max_depth - 1):
        for i in range(n):
            for j in occ_others(occ1, text, i):
                lits.append(lm.new_id(lm.lit.link_to, depth, i, j))
    for i in range(n):
        lits.append(lm.new_id(lm.lit.fbeg, i))
        lits.append(lm.new_id(lm.lit.root, i))
    for i in range(n):
        for j in occ_others(occ1, text, i):
            lits.append(lm.new_id(lm.lit.ref, i, j))
    for depth in range(max_depth - 1):
        for i in range(n):
            lits.append(lm.new_id(lm.lit.depth_ref, depth, i))
    wcnf.append(lits)

    # set objective to minimizes the number of factors
    for i in range(n):
        fbeg0 = lm.getid(lm.lit.fbeg, i)
        wcnf.append([-fbeg0], weight=1)

    # if text[i:i+2] occurs once, i+1 is the beginning of a factor
    count = 0
    for i in range(n - 1):
        if len(occ2[text[i : i + 2]]) == 1:
            fbeg = lm.getid(lm.lit.fbeg, i + 1)
            wcnf.append([fbeg])
            count += 1
    logger.info(f"{count}/{n} occurs only once")

    # if i is a root, i+1 is the beginning of a factor
    for i in range(n):
        root0 = lm.getid(lm.lit.root, i)
        fbeg0 = lm.getid(lm.lit.fbeg, i)
        wcnf.append(pysat_if(root0, fbeg0))
        if i + 1 < n:
            fbeg1 = lm.getid(lm.lit.fbeg, i + 1)
            wcnf.append(pysat_if(root0, fbeg1))

    # relation between link and ref
    # link i to j implies ref j to i.
    for i in range(n):
        for j in occ_others(occ1, text, i):
            assert 0 <= i, j < n
            ref_ji0 = lm.getid(lm.lit.ref, j, i)
            fbeg_j = lm.getid(lm.lit.fbeg, j)
            for depth in range(max_depth - 1):
                link_ij = lm.getid(lm.lit.link_to, depth, i, j)
                # if there is link i to j, there is ref j to i.
                wcnf.append(pysat_if(link_ij, ref_ji0))
            if i == 0 or j == 0 or text[i - 1] != text[j - 1]:
                # if ref j to i and their previous positions do not match,
                # j is the beginning of a factor
                wcnf.append(pysat_if(ref_ji0, fbeg_j))
            if i > 0 and j > 0 and text[i - 1] == text[j - 1]:
                ref_ji1 = lm.getid(lm.lit.ref, j - 1, i - 1)
                fbeg0 = lm.getid(lm.lit.fbeg, j)
                # here, text[i-1:i+1] == text[j-1:j+1]
                # if ref j-1 does not refer to i-1, j is the beginning of a factor
                wcnf.append(pysat_if_and_then_or([-ref_ji1, ref_ji0], [fbeg0]))
                # if ref j-1 refer to i-1, j is not the beginning of a factor
                # this is because if not, the result size is not the minimum.
                wcnf.append(pysat_if_and_then_or([ref_ji1, ref_ji0], [-fbeg0]))

    # if there is a link i to j at depth>1, there must exist a link k to j at depth-1.
    logger.debug("link is connectted, and it forms tree structure")
    for depth in range(1, max_depth - 1):
        if depth % 30 == 0:
            logger.debug(f"depth {depth}/{max_depth}")
        for i in range(n):
            for j in occ_others(occ1, text, i):
                link_ij = lm.getid(lm.lit.link_to, depth, i, j)
                dref_i = lm.getid(lm.lit.depth_ref, depth - 1, i)
                wcnf.append(pysat_if(link_ij, dref_i))

    # roots for each character are only one.
    # this is not necessity, it may cause bad effect.
    for c in occ1.keys():
        roots = [lm.getid(lm.lit.root, i) for i in occ1[c]]
        wcnf.extend(pysat_equal(lm, 1, roots))
        # wcnf.append(pysat_atleast_one(roots))

    # the number of refferences for each position is at most one.
    # root position does not refer to any positions.
    logger.debug("# of referrences is only one")
    for i in range(n):
        refs = [lm.getid(lm.lit.ref, i, j) for j in occ1[text[i]] if i != j]
        root_i = lm.getid(lm.lit.root, i)
        # the number of refferences for each position is at most one.
        wcnf.extend(CardEnc.atmost(refs, bound=1, vpool=lm.vpool))
        # wcnf.extend(pysat_equal(lm, 1, refs + [root_i]))
        for j in occ_others(occ1, text, i):
            ref_ij = lm.getid(lm.lit.ref, i, j)
            link_ij = lm.getid(lm.lit.link_to, 0, i, j)
            # root position does not refer to any positions.
            wcnf.append(pysat_if(root_i, -ref_ij))
            wcnf.append(pysat_if(-root_i, -link_ij))

    # this is the most heavy process
    # the number of links to j is at most one.
    # define depth ref
    logger.debug("each position, it has only one link or root")
    # for i in range(n):
    #     rooti = lm.getid(lm.lit.root, i)
    #     links_i = [
    #         lm.getid(lm.lit.link_to, depth, j, i)
    #         for depth in range(max_depth - 1)
    #         for j in occ1[text[i]]
    #         if i != j
    #     ]
    #     wcnf.extend(pysat_equal(lm, 1, links_i + [rooti]))
    for depth in range(max_depth - 1):
        if depth % 30 == 0:
            logger.debug(f"depth {depth}/{max_depth}")
        for i in range(n):
            for j in occ_others(occ1, text, i):
                link_ij = lm.getid(lm.lit.link_to, depth, i, j)
                dref_j = lm.getid(lm.lit.depth_ref, depth, j)
                # if there exists link to j, dref to j is true
                wcnf.append(pysat_if(link_ij, dref_j))
            links_to_i = [
                lm.getid(lm.lit.link_to, depth, j, i) for j in occ1[text[i]] if i != j
            ]
            dref_i = lm.getid(lm.lit.depth_ref, depth, i)
            # if dref to i is true, there is (at least one) link to i.
            wcnf.append(pysat_if_and_then_or([dref_i], links_to_i))

            # there is at most one link to i at each depth.
            wcnf.extend(CardEnc.atmost(links_to_i, bound=1, vpool=lm.vpool))

            # if dref to i is false, every link to i is false
            if_then, clauses = pysat_and(lm.new_id, [-x for x in links_to_i])
            wcnf.extend(clauses)
            wcnf.append(pysat_if(-dref_i, if_then))
    # for each position, there must exist either a root or reference to a position
    for i in range(n):
        dref_i = [
            lm.getid(lm.lit.depth_ref, depth, i) for depth in range(max_depth - 1)
        ]
        root_i = lm.getid(lm.lit.root, i)
        wcnf.extend(pysat_equal(lm, 1, dref_i + [root_i]))

    logger.info(
        f"#literals = {lm.top()}, # hard clauses={len(wcnf.hard)}, # of soft clauses={len(wcnf.soft)}"
    )
    return lm, wcnf


def encode(text: bytes) -> BiDirType:
    lm, wcnf = bidirectional_WCNF(text)
    solver = RC2(wcnf)
    sol = solver.compute()
    assert sol is not None
    sold = dict()
    for x in sol:
        sold[abs(x)] = x > 0
    factors = sol2bidirectional(lm, sold, text)
    return factors


def bd_assumptions(lm: BiDirLiteralManager, factors: BiDirType) -> list[list[int]]:
    """
    returns assumption of given bidirectional scheme.
    """
    i = 0
    res = []
    use = [True for _ in range(len(text))]
    for j in range(len(text)):
        use[j] = False
    for f in factors:
        logger.debug(f"factors[{i}]={f}")
        if f[0] == -1:
            if use[i]:
                res.append([lm.getid(lm.lit.fbeg, i)])
                res.append([lm.getid(lm.lit.root, i)])
            i += 1
        else:
            if use[i]:
                res.append([lm.getid(lm.lit.fbeg, i)])
                res.append([lm.getid(lm.lit.ref, i, f[0])])
            for j in range(1, f[1]):
                if use[i + j]:
                    res.append([-lm.getid(lm.lit.fbeg, i + j)])
                    res.append([lm.getid(lm.lit.ref, i + j, f[0] + j)])
            i += f[1]
    return res


def bidirectional(text: bytes, exp: BiDirExp = None):
    total_start = time.time()
    lm, wcnf = bidirectional_WCNF(text)
    for lname in lm.nvar.keys():
        logger.info(f"# of [{lname}] literals  = {lm.nvar[lname]}")

    if exp:
        exp.time_prep = time.time() - total_start

    # solver = RC2(wcnf, trim=10, verbose=3)
    solver = RC2(wcnf)
    sol = solver.compute()

    assert sol is not None
    sold = dict()
    for x in sol:
        sold[abs(x)] = x > 0

    def show_lits(lits):
        for lit in sorted(lits):
            if lit[1]:
                logger.debug(lit[0])

    # show_lits(sol2lits2(lm, sold, lm.lit.link_to))
    # show_lits(sol2lits2(lm, sold, lm.lit.ref))
    # show_lits(sol2lits2(lm, sold, lm.lit.fbeg))
    # show_lits(sol2lits2(lm, sold, lm.lit.root))
    show_sol(lm, sold, text)
    factors = sol2bidirectional(lm, sold, text)

    logger.debug(factors)
    logger.debug(f"original={text}")
    logger.debug(f"decode={decode(factors)}")
    assert decode(factors) == text
    if exp:
        exp.time_total = time.time() - total_start
        exp.sol_nvars = wcnf.nv
        exp.sol_nhard = len(wcnf.hard)
        exp.sol_nsoft = len(wcnf.soft)
        exp.bd_factors = factors
        exp.bd_size = len(factors)
    return factors


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum Bidirectional Scheme")
    parser.add_argument("--file", type=str, help="input file", default="")
    # parser.add_argument("--output", type=str, help="output file", default="")

    args = parser.parse_args()
    if args.file == "":
        parser.print_help()
        sys.exit()
    return args


if __name__ == "__main__":
    args = parse_args()
    text = open(args.file, "rb").read()
    logger.info(text)

    timer = Timer()

    # text = b"ababab"
    # text = b"abbbaaabb"
    # text = b"abbbaaabb"
    # exp = BiDirExp()
    exp = BiDirExp.create()
    exp.algo = "solver"
    exp.file_name = os.path.basename(args.file)
    exp.file_len = len(text)
    factors_sol = bidirectional(text, exp)
    exp.bd_factors = factors_sol
    exp.bd_size = len(factors_sol)
    logger.info(f"runtime: {timer()}")
    # logger.info("solver factors")
    logger.info(bd_info(factors_sol, text))
    print(exp.to_json(ensure_ascii=False))  # type: ignore
    # assert len(factors_sol) == 4

    # assert len(factors_sol) == len(factors_hdbn)
