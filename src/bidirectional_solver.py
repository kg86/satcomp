# compute the smallest bidirectional macro scheme by using SAT solver
from enum import auto
import json
import sys
import argparse
import os
from logging import getLogger, DEBUG, INFO, CRITICAL, StreamHandler, Formatter
import time
from typing import Dict, Iterator, List, Optional

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
logger.addHandler(handler)


class BiDirLiteral(Enum):
    true = Literal.true
    false = Literal.false
    auxlit = Literal.auxlit
    depth_ref = auto()
    root = auto()
    fbeg = auto()
    ref = auto()
    any_ref = auto()


class BiDirLiteralManager(LiteralManager):
    """
    Manage literals used for solvers.
    """

    def __init__(self, text: bytes, max_depth: int):
        self.text = text
        self.max_depth = max_depth
        self.n = len(self.text)
        self.lits = BiDirLiteral
        self.verifyf = {
            BiDirLiteral.depth_ref: self.verify_link,
            BiDirLiteral.root: self.verify_root,
            BiDirLiteral.ref: self.verify_ref,
            BiDirLiteral.any_ref: self.verify_depth_ref,
        }
        super().__init__(self.lits)

    def newid(self, *obj) -> int:
        res = super().newid(*obj)
        if len(obj) > 0 and obj[0] in self.verifyf:
            self.verifyf[obj[0]](obj)
        return res

    def verify_link(self, obj: Tuple[str, int, int, int]):
        # obj = (name, depth, form, to)
        assert len(obj) == 4
        assert obj[0] == self.lits.depth_ref
        assert 0 <= obj[1] < self.max_depth - 1
        assert 0 <= obj[2], obj[3] < self.n
        assert obj[2] != obj[3]
        assert self.text[obj[2]] == self.text[obj[3]]

    def verify_root(self, obj: Tuple[str, int]):
        # obj = (name, pos)
        assert len(obj) == 2
        assert 0 <= obj[1] < self.n

    def verify_ref(self, obj: Tuple[str, int, int]):
        # obj = (name, pos, ref_pos)
        assert len(obj) == 3
        assert obj[1] != obj[2]
        assert 0 <= obj[1], obj[2] < self.n
        assert self.text[obj[1]] == self.text[obj[2]]

    def verify_depth_ref(self, obj: Tuple[str, int, int]):
        # obj = (name, depth, ref_pos)
        assert len(obj) == 3
        assert 0 <= obj[1] < self.max_depth
        assert 0 <= obj[2] < self.n


def pysat_equal(lm: BiDirLiteralManager, bound: int, lits: List[int]):
    return CardEnc.equals(lits, bound=bound, encoding=EncType.pairwise, vpool=lm.vpool)


def sol2lits(lm: BiDirLiteralManager, sol: Dict[int, bool], lit_name: str) -> list:
    """
    Transform the result of the sat solver to literals of name `lit_name`.
    Returns the list whose element is formed of (literal, True or False)
    """
    res = []
    for id, obj in lm.vpool.id2obj.items():
        assert isinstance(id, int)
        if obj[0] == lit_name:
            res.append((obj, sol[id]))

    return res


def sol2refs(lm: BiDirLiteralManager, sol: Dict[int, bool], text: bytes):
    """
    Reference dictionary refs[i] = j s.t. position i refers to position j.
    """
    n = len(text)
    occ = make_occa1(text)
    refs = dict()
    for i in range(n):
        for j in occ[text[i]]:
            if i == j:
                continue
            if sol[lm.getid(lm.lits.ref, i, j)]:
                refs[i] = j
                break
    logger.debug(f"refs={refs}")
    return refs


def show_sol(lm: BiDirLiteralManager, sol: Dict[int, bool], text: bytes):
    """
    Show the result of SAT solver.
    """
    n = len(text)
    occ = make_occa1(text)
    pinfo = defaultdict(list)

    def refer_to(i: int) -> List[int]:
        res = []
        for depth in range(lm.max_depth - 1):
            for j in occ[text[i]]:
                if i == j:
                    continue
                if sol[lm.getid(lm.lits.depth_ref, depth, j, i)]:
                    res.append((lm.lits.depth_ref, depth, j, i))
        return res

    for i in range(n):
        pinfo[i].append(chr(text[i]))
        for j in occ_others(occ, text, i):
            key = (lm.lits.ref, i, j)
            if sol[lm.getid(*key)]:
                pinfo[i].append(str(key))
        key = (lm.lits.root, i)
        lid = lm.getid(*key)
        if sol[lid]:
            pinfo[i].append(str(key))
        fbeg_key = (lm.lits.fbeg, i)
        if sol[lm.getid(*fbeg_key)]:
            pinfo[i].append(str(fbeg_key))
        for key in refer_to(i):
            pinfo[i].append(f"{key}")
    for i in range(n):
        logger.debug(f"i={i} " + ", ".join(pinfo[i]))


def sol2bidirectional(
    lm: BiDirLiteralManager, sol: Dict[int, bool], text: bytes
) -> BiDirType:
    """
    Compute bidirectional macro schemes from the result of SAT solver.
    """
    res = BiDirType([])
    fbegs = []
    n = len(text)
    refs = sol2refs(lm, sol, text)
    for i in range(n):
        if sol[lm.getid(lm.lits.fbeg, i)]:
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


def make_occa1(text: bytes) -> Dict[int, List[int]]:
    """
    occurrences of characters
    """
    occ = defaultdict(list)
    for i in range(len(text)):
        occ[text[i]].append(i)
    return occ


def make_occa2(text: bytes) -> Dict[bytes, List[int]]:
    """
    occurrences of length-2 substrings
    """
    match2 = defaultdict(list)
    for i in range(len(text) - 1):
        match2[text[i : i + 2]].append(i)
    return match2


def occ_others(occ1: Dict[int, List[int]], text: bytes, i: int):
    """
    returns occurrences of `text[i]` in `text` except `i`.
    """
    for j in occ1[text[i]]:
        if i != j:
            yield j


def bidirectional_WCNF(text: bytes) -> Tuple[BiDirLiteralManager, WCNF]:
    """
    Compute the max sat formula for computing the smallest bidirectional macro schemes.
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
    wcnf.append([lm.getid(lm.lits.true)])
    wcnf.append([lm.getid(lm.lits.false)])

    # register all literals (except auxiliary literals) to literal manager
    lits = [lm.sym2id(lm.true)]
    for depth in range(max_depth - 1):
        for i in range(n):
            for j in occ_others(occ1, text, i):
                # depth_ref(depth, i, j) is true iff i refers to j at depth
                lits.append(lm.newid(lm.lits.depth_ref, depth, i, j))
    for i in range(n):
        # fbeg(i) is true iff a factor begins at i
        lits.append(lm.newid(lm.lits.fbeg, i))
        # root(i) is true iff a factor at i represents a single character not a reference
        lits.append(lm.newid(lm.lits.root, i))
    for i in range(n):
        for j in occ_others(occ1, text, i):
            # ref(i, j) is true iff i refers to j
            lits.append(lm.newid(lm.lits.ref, i, j))
    for depth in range(max_depth - 1):
        for i in range(n):
            # any_ref(depth, i) is true iff i refers to any position at depth
            lits.append(lm.newid(lm.lits.any_ref, depth, i))
    wcnf.append(lits)

    # objective: minimizes the number of factors
    for i in range(n):
        fbeg0 = lm.getid(lm.lits.fbeg, i)
        wcnf.append([-fbeg0], weight=1)

    # objective to run fast: if text[i:i+2] occurs only once, i+1 is the beginning of a factor
    count = 0
    for i in range(n - 1):
        if len(occ2[text[i : i + 2]]) == 1:
            fbeg = lm.getid(lm.lits.fbeg, i + 1)
            wcnf.append([fbeg])
            count += 1
    logger.info(f"{count}/{n} occurs only once")

    # objective: valid references

    # the following is the most heavy process
    logger.debug("each position, it has only one link or root")
    for depth in range(max_depth - 1):
        if depth % 30 == 0:
            logger.debug(f"depth {depth}/{max_depth}")
        for i in range(n):
            for j in occ_others(occ1, text, i):
                dref_ji = lm.getid(lm.lits.depth_ref, depth, j, i)
                dref_j = lm.getid(lm.lits.any_ref, depth, j)
                # tree-1: if j refers to i at depth, j refers to any position at depth
                # this is the definition of any_ref(depth, j)
                wcnf.append(pysat_if(dref_ji, dref_j))
            refi = [
                lm.getid(lm.lits.depth_ref, depth, i, j)
                for j in occ1[text[i]]
                if i != j
            ]
            if refi:
                dref_i = lm.getid(lm.lits.any_ref, depth, i)
                # tree-2: if i refers to any position at depth, there is a reference from i to j
                wcnf.append(pysat_if_and_then_or([dref_i], refi))

                # tree-3: the number of references from i is at most one
                wcnf.extend(CardEnc.atmost(refi, bound=1, vpool=lm.vpool))

                no_refi, clauses = pysat_and(lm.newid, [-x for x in refi])
                wcnf.extend(clauses)
                # tree-4: if i does not refer to any position at depth, there is no references from i
                wcnf.append(pysat_if(-dref_i, no_refi))
    for i in range(n):
        dref_i = [lm.getid(lm.lits.any_ref, depth, i) for depth in range(max_depth - 1)]
        root_i = lm.getid(lm.lits.root, i)
        # tree-5: each position is a root or has a reference to any position
        # (use all positions)
        wcnf.extend(pysat_equal(lm, 1, dref_i + [root_i]))
    for c in occ1.keys():
        roots = [lm.getid(lm.lits.root, i) for i in occ1[c]]
        # a root for each character exists only one.
        # this is not necessity, it may cause bad effect.
        wcnf.extend(pysat_equal(lm, 1, roots))
        # wcnf.append(pysat_atleast_one(roots))

    for depth in range(1, max_depth - 1):
        if depth % 30 == 0:
            logger.debug(f"depth {depth}/{max_depth}")
        for i in range(n):
            for j in occ_others(occ1, text, i):
                dref_ji = lm.getid(lm.lits.depth_ref, depth, j, i)
                dref_i = lm.getid(lm.lits.any_ref, depth - 1, i)
                # tree-5: if j refers to j at depth, i refers to any position at dpeth-1
                wcnf.append(pysat_if(dref_ji, dref_i))
    # ----------- end of valid reference ----
    # bridge
    for i in range(n):
        for j in occ_others(occ1, text, i):
            assert 0 <= i, j < n
            ref_ji0 = lm.getid(lm.lits.ref, j, i)
            fbeg_j = lm.getid(lm.lits.fbeg, j)
            for depth in range(max_depth - 1):
                dref_ji = lm.getid(lm.lits.depth_ref, depth, j, i)
                # bridge-1: if j refers to i at depth, j refers to i
                wcnf.append(pysat_if(dref_ji, ref_ji0))
            if i == 0 or j == 0 or text[i - 1] != text[j - 1]:
                # bridge-2: since it is impossible refer to i-1 or j-1, factor begins at j.
                wcnf.append(pysat_if(ref_ji0, fbeg_j))
            if i > 0 and j > 0 and text[i - 1] == text[j - 1]:
                ref_ji1 = lm.getid(lm.lits.ref, j - 1, i - 1)
                fbeg0 = lm.getid(lm.lits.fbeg, j)
                # here, text[i-1:i+1] == text[j-1:j+1]
                # bridge-3: if j-1 does not refer to i-1 and j refers to i, factor begins at j
                wcnf.append(pysat_if_and_then_or([-ref_ji1, ref_ji0], [fbeg0]))
                # bridge-4: if j-1 and j refer to i-1 and i, respectively, factor does not begin at j
                # because if not, the result size is not the minimum.
                wcnf.append(pysat_if_and_then_or([ref_ji1, ref_ji0], [-fbeg0]))

    logger.debug("# of referrences is only one")
    for i in range(n):
        refs = [lm.getid(lm.lits.ref, i, j) for j in occ1[text[i]] if i != j]
        root_i = lm.getid(lm.lits.root, i)
        # the number of rerferences from a position is at most one.
        wcnf.extend(CardEnc.atmost(refs, bound=1, vpool=lm.vpool))
        # wcnf.extend(pysat_equal(lm, 1, refs + [root_i]))
        for j in occ_others(occ1, text, i):
            ref_ij = lm.getid(lm.lits.ref, i, j)
            dref_ji = lm.getid(lm.lits.depth_ref, 0, j, i)
            # root position does not refer to any positions.
            wcnf.append(pysat_if(root_i, -ref_ij))
            # any non root position is not refered from any positions
            wcnf.append(pysat_if(-root_i, -dref_ji))

    logger.info(
        f"#literals = {lm.top()}, # hard clauses={len(wcnf.hard)}, # of soft clauses={len(wcnf.soft)}"
    )

    # objective: bridge objectives between beginning factor and valid references
    for i in range(n):
        root0 = lm.getid(lm.lits.root, i)
        fbeg0 = lm.getid(lm.lits.fbeg, i)
        wcnf.append(pysat_if(root0, fbeg0))
        if i + 1 < n:
            fbeg1 = lm.getid(lm.lits.fbeg, i + 1)
            # if i is root, a factor begins at i
            wcnf.append(pysat_if(root0, fbeg1))

    return lm, wcnf


def min_bidirectional(text: bytes, exp: Optional[BiDirExp] = None) -> BiDirType:
    """
    Compute the smallest bidirectional macro schemes.
    """
    total_start = time.time()
    lm, wcnf = bidirectional_WCNF(text)
    for lname in lm.nvar.keys():
        logger.info(f"# of [{lname}] literals  = {lm.nvar[lname]}")

    if exp:
        exp.time_prep = time.time() - total_start

    # solver = RC2(wcnf, verbose=3)
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

    # show_lits(sol2lits2(lm, sold, lm.lit.ref))
    show_sol(lm, sold, text)
    factors = sol2bidirectional(lm, sold, text)

    logger.debug(factors)
    logger.debug(f"original={text}")
    logger.debug(f"decode={decode(factors)}")
    assert decode(factors) == text
    if exp:
        exp.time_total = time.time() - total_start
        exp.factors = factors
        exp.factor_size = len(factors)
        exp.fill(wcnf)
    return factors


def get_sold(sol: List[int]):
    """
    Compute dictionary res[literal_id] = True or False.
    """
    sold = dict()
    for x in sol:
        sold[abs(x)] = x > 0
    return sold


def bidirectional_enumerate(text: bytes) -> Iterator[BiDirType]:
    lm, wcnf = bidirectional_WCNF(text)
    solset = set()
    overlap = 0
    with RC2(wcnf) as solver:
        for sol in solver.enumerate():
            factors = sol2bidirectional(lm, get_sold(sol), text)
            key = tuple(factors)
            if key not in solset:
                solset.add(key)
                logger.info(f"overlap solution = {overlap}")
                yield factors
            else:
                overlap += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum Bidirectional Scheme")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument(
        "--log_level",
        type=str,
        help="log level, DEBUG/INFO/CRITICAL",
        default="CRITICAL",
    )

    args = parser.parse_args()
    if (args.file == "" and args.str== "") or (
        args.log_level not in ["DEBUG", "INFO", "CRITICAL"]
    ):
        parser.print_help()
        sys.exit()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.str != "":
        text = args.str.encode("utf8")
    else:
        text = open(args.file, "rb").read()

    if args.log_level == "DEBUG":
        logger.setLevel(DEBUG)
    elif args.log_level == "INFO":
        logger.setLevel(INFO)
    elif args.log_level == "CRITICAL":
        logger.setLevel(CRITICAL)

    logger.info(text)

    timer = Timer()

    exp = BiDirExp.create()
    exp.algo = "bidirectional-sat"
    exp.file_name = os.path.basename(args.file)
    exp.file_len = len(text)
    factors_sol = min_bidirectional(text, exp)
    exp.factors = factors_sol
    exp.factor_size = len(factors_sol)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
