# compute the smallest bidirectional macro scheme by using SAT solver
# version that reduces total CNF size to O(N^3)

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from enum import auto
from logging import CRITICAL, DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import Dict, Iterator, List, Optional, Tuple

from pysat.card import CardEnc
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

import lz77
from bidirectional import BiDirExp, BiDirType, decode
from mysat import Enum, Literal, LiteralManager, pysat_and, pysat_if, pysat_or
from mytimer import Timer

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
    pstart = auto()  # i: true iff T[i] is start of phrase
    root = auto()  # i: true iff T[i] is ground phrase
    ref = auto()  # (i,j) true iff position T[i] references position T[j]
    tref = (
        auto()
    )  # (i,j) true iff position T[i] eventually references position T[j] (transitive closure)


class BiDirLiteralManager(LiteralManager):
    """
    Manage literals used for solvers.
    """

    def __init__(self, text: bytes):
        self.text = text
        self.n = len(self.text)
        self.lits = BiDirLiteral
        self.verifyf = {
            BiDirLiteral.root: self.verify_root,
            BiDirLiteral.ref: self.verify_pstart,
            BiDirLiteral.ref: self.verify_ref,
            BiDirLiteral.ref: self.verify_tref,
        }
        super().__init__(self.lits)

    def newid(self, *obj) -> int:
        res = super().newid(*obj)
        if len(obj) > 0 and obj[0] in self.verifyf:
            self.verifyf[obj[0]](obj)
        return res

    def verify_root(self, obj: Tuple[str, int]):
        # obj = (name, pos)
        assert len(obj) == 2
        assert 0 <= obj[1] < self.n
        pass

    def verify_pstart(self, obj: Tuple[str, int]):
        # obj = (name, pos)
        assert len(obj) == 2
        assert 0 <= obj[1] < self.n
        pass

    def verify_ref(self, obj: Tuple[str, int, int]):
        # obj = (name, pos, ref_pos)
        assert len(obj) == 3
        assert obj[1] != obj[2]
        assert 0 <= obj[1], obj[2] < self.n
        assert self.text[obj[1]] == self.text[obj[2]]

    def verify_tref(self, obj: Tuple[str, int, int]):
        # obj = (name, pos, ref_pos)
        assert len(obj) == 3
        assert obj[1] != obj[2]
        assert 0 <= obj[1], obj[2] < self.n
        assert self.text[obj[1]] == self.text[obj[2]]


def pysat_equal(lm: BiDirLiteralManager, bound: int, lits: List[int]):
    return CardEnc.equals(lits, bound=bound, vpool=lm.vpool)


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
        fbeg_key = (lm.lits.pstart, i)
        if sol[lm.getid(*fbeg_key)]:
            pinfo[i].append(str(fbeg_key))

        for j in occ_others(occ, text, i):
            key = (lm.lits.tref, i, j)
            if sol[lm.getid(*key)]:
                pinfo[i].append(str(key))
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
        if sol[lm.getid(lm.lits.pstart, i)]:
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

    lm = BiDirLiteralManager(text)
    wcnf = WCNF()

    # register all literals (except auxiliary literals) to literal manager
    # lits = [lm.sym2id(lm.true)]
    lits = []
    for i in range(n):
        # pstart(i) is true iff a factor begins at i
        lits.append(lm.newid(lm.lits.pstart, i))
        # root(i) is true iff T[i] is a ground phrase
        lits.append(lm.newid(lm.lits.root, i))
    for i in range(n):
        for j in occ_others(occ1, text, i):
            # ref(i, j) is true iff i refers to j
            lits.append(lm.newid(lm.lits.ref, i, j))
            # tref(i, j) is true iff i eventualy refers to j
            lits.append(lm.newid(lm.lits.tref, i, j))
    ############################################################################

    logger.debug("each position has exactly one reference, or is a root")
    for i in range(n):
        ref_or_root = [
            lm.getid(lm.lits.ref, i, j) for j in occ_others(occ1, text, i)
        ] + [lm.getid(lm.lits.root, i)]
        wcnf.extend(pysat_equal(lm, 1, ref_or_root))

    for c in occ1.keys():
        for i in occ1[c]:
            for j in occ_others(occ1, text, i):
                # if ref(i,j) -> tref(i,j)
                wcnf.append(
                    [-lm.getid(lm.lits.ref, i, j), lm.getid(lm.lits.tref, i, j)]
                )
                for k in occ1[c]:
                    if i != k and j != k:
                        wcnf.append(  # if tref(i,k) and ref(k,j) -> tref(i,j)
                            [
                                -lm.getid(lm.lits.tref, i, k),
                                -lm.getid(lm.lits.ref, k, j),
                                lm.getid(lm.lits.tref, i, j),
                            ]
                        )
    for c in occ1.keys():
        for i in occ1[c]:
            for j in occ_others(occ1, text, i):
                # if tref(i,j) then (ref(i,j) or there exists k s.t. tref(i,k) and ref(k,j))
                # not tref(i,j) or ref(i,j) or exists k s.t. (tref(i,k) and ref(k,j))
                pred = []
                for k in occ_others(occ1, text, i):
                    if k != j:
                        pred1, clauses = pysat_and(
                            lm.newid,
                            (
                                [
                                    lm.getid(lm.lits.tref, i, k),
                                    lm.getid(lm.lits.ref, k, j),
                                ]
                            ),
                        )
                        wcnf.extend(clauses)
                        pred.append(pred1)
                wcnf.append(
                    [-lm.getid(lm.lits.tref, i, j)]
                    + [lm.getid(lm.lits.ref, i, j)]
                    + pred
                )

    for c in occ1.keys():
        for i in occ1[c]:
            # each postion i is either a root(i) or there exist j s.t. (tref(i,j) and root(j))
            reach_j = []
            for j in occ_others(occ1, text, i):
                reach, clauses = pysat_and(
                    lm.newid,
                    [lm.getid(lm.lits.tref, i, j), lm.getid(lm.lits.root, j)],
                )
                wcnf.extend(clauses)
                reach_j.append(reach)
            wcnf.append([lm.getid(lm.lits.root, i)] + reach_j)

    # a root must be a beginning of a phrase: root(i) -> pstart(i)
    for i in range(n):
        root0 = lm.getid(lm.lits.root, i)
        fbeg0 = lm.getid(lm.lits.pstart, i)
        wcnf.append(pysat_if(root0, fbeg0))

    # if i = 0 or j = 0 or T[i-1] \neq T[j-1]: not (ref(i,j)) or pstart(i)
    for c in occ1.keys():
        for i in occ1[c]:
            for j in occ_others(occ1, text, i):
                if i == 0 or j == 0 or text[i - 1] != text[j - 1]:
                    wcnf.append(
                        [-lm.getid(lm.lits.ref, i, j), lm.getid(lm.lits.pstart, i)]
                    )
    # for i,j > 0, and T[i] = T[j], T[i-1] = T[j-1]
    # if (root(i-1) or not (ref(i-1,j-1)) and ref(i,j)) => pstart(i)
    # (not (root(i-1)) and (ref(i-1,j-1) or not (ref(i,j)))) or pstart(i)
    for c in occ1.keys():
        for i in occ1[c]:
            for j in occ_others(occ1, text, i):
                if i > 0 and j > 0 and text[i - 1] == text[j - 1]:
                    pred1, clauses = pysat_and(
                        lm.newid,
                        [
                            -lm.getid(lm.lits.ref, i - 1, j - 1),
                            lm.getid(lm.lits.ref, i, j),
                        ],
                    )
                    wcnf.extend(clauses)
                    pred2, clauses = pysat_or(
                        lm.newid, [lm.getid(lm.lits.root, i - 1), pred1]
                    )
                    wcnf.extend(clauses)
                    wcnf.append([-pred2, lm.getid(lm.lits.pstart, i)])

    # the first position is always a beginning of a phrase
    wcnf.append([lm.getid(lm.lits.pstart, 0)])

    # objective: minimizes the number of factors
    for i in range(n):
        wcnf.append([-lm.getid(lm.lits.pstart, i)], weight=1)

    return lm, wcnf


def min_bidirectional(
    text: bytes, exp: Optional[BiDirExp] = None, contain_list: List[int] = []
) -> BiDirType:
    """
    Compute the smallest bidirectional macro schemes.
    """
    total_start = time.time()
    lm, wcnf = bidirectional_WCNF(text)
    for lname in lm.nvar.keys():
        logger.info(f"# of [{lname}] literals  = {lm.nvar[lname]}")

    for i in contain_list:
        fbeg0 = lm.getid(lm.lits.pstart, i)
        wcnf.append([fbeg0])

    if exp:
        exp.time_prep = time.time() - total_start

    # solver = RC2(wcnf, verbose=3)
    solver = RC2(wcnf)
    sol = solver.compute()

    assert sol is not None
    sold = get_sold(sol)

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
        "--contains",
        nargs="+",
        type=int,
        help="list of text positions that must be a beginning of a phrase, starting with index 0",
        default=[],
    )
    parser.add_argument(
        "--log_level",
        type=str,
        help="log level, DEBUG/INFO/CRITICAL",
        default="CRITICAL",
    )

    args = parser.parse_args()
    if (args.file == "" and args.str == "") or (
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
    factors_sol = min_bidirectional(text, exp, args.contains)
    exp.factors = factors_sol
    exp.factor_size = len(factors_sol)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
