# compute the smallest bidirectional macro scheme (BMS) by using SAT solver
# Original version described in the ESA paper
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
from pysat.formula import CNF, WCNF

import lz77
from bms import BiDirExp, BiDirType, decode
from mysat import (
    Enum,
    Literal,
    LiteralManager,
)
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
    ref = auto()  # (i,j) true iff position T[i] references position T[j]
    tref = auto()  # (i,j) true iff position T[i] eventually references position T[j] (transitive closure)


class BiDirLiteralManager(LiteralManager):
    """
    Manage literals used for solvers.
    """

    def __init__(self, text: bytes):
        self.text = text
        self.n = len(self.text)
        super().__init__()

    def add_pstart(self, pos: int) -> None:
        assert 0 <= pos < self.n
        self.newid(BiDirLiteral.pstart, pos)

    def add_ref(self, pos: int, ref_pos: int) -> None:
        assert pos != ref_pos
        assert 0 <= pos < self.n
        assert 0 <= ref_pos < self.n
        assert self.text[pos] == self.text[ref_pos]
        self.newid(BiDirLiteral.ref, pos, ref_pos)

    def add_tref(self, pos: int, ref_pos: int) -> None:
        assert pos != ref_pos
        assert 0 <= pos < self.n
        assert 0 <= ref_pos < self.n
        assert self.text[pos] == self.text[ref_pos]
        self.newid(BiDirLiteral.tref, pos, ref_pos)


def pysat_equal(lm: BiDirLiteralManager, bound: int, lits: List[int]) -> CNF:
    return CardEnc.equals(lits, bound=bound, vpool=lm.vpool)


def sol2refs(lm: BiDirLiteralManager, sol: Dict[int, bool], text: bytes) -> Dict[int, int]:
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
            if sol[lm.getid(BiDirLiteral.ref, i, j)]:
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
            key = (BiDirLiteral.ref, i, j)
            if sol[lm.getid(*key)]:
                pinfo[i].append(str(key))
        fbeg_key = (BiDirLiteral.pstart, i)
        if sol[lm.getid(*fbeg_key)]:
            pinfo[i].append(str(fbeg_key))
        for j in occ_others(occ, text, i):
            key = (BiDirLiteral.tref, i, j)
            if sol[lm.getid(*key)]:
                pinfo[i].append(str(key))
    for i in range(n):
        logger.debug(f"i={i} " + ", ".join(pinfo[i]))


def sol2bms(lm: BiDirLiteralManager, sol: Dict[int, bool], text: bytes) -> BiDirType:
    """
    Compute a bidirectional macro scheme (BMS) from the result of SAT solver.
    """
    res = BiDirType([])
    fbegs = []
    n = len(text)
    refs = sol2refs(lm, sol, text)
    for i in range(n):
        if sol[lm.getid(BiDirLiteral.pstart, i)]:
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


def bms_WCNF(text: bytes) -> Tuple[BiDirLiteralManager, WCNF]:
    """
    Compute the max sat formula for computing the smallest bidirectional macro scheme (BMS).
    """
    n = len(text)
    lz77fs = lz77.encode(text)
    logger.info("bms_solver start")
    logger.info(f"# of text = {n}, # of lz77 = {len(lz77fs)}")

    occ1 = make_occa1(text)

    lm = BiDirLiteralManager(text)
    wcnf = WCNF()

    # register all literals (except auxiliary literals) to literal manager
    lits = []
    for i in range(n):
        # pstart(i) is true iff a factor begins at i
        lits.append(lm.add_pstart(i))
    for i in range(n):
        for j in occ_others(occ1, text, i):
            # ref(i, j) is true iff i refers to j
            lits.append(lm.add_ref(i, j))
            # tref(i, j) is true iff i eventualy refers to j
            lits.append(lm.add_tref(i, j))
    ############################################################################
    logger.debug("each position has atmost one reference")
    for i in range(n):
        refi = [lm.getid(BiDirLiteral.ref, i, j) for j in occ_others(occ1, text, i)]
        wcnf.extend(CardEnc.atmost(refi, bound=1, vpool=lm.vpool))

    for c in occ1.keys():
        for i in occ1[c]:
            for j in occ_others(occ1, text, i):
                # if ref(i,j) -> tref(i,j)
                wcnf.append([-lm.getid(BiDirLiteral.ref, i, j), lm.getid(BiDirLiteral.tref, i, j)])
                for k in occ1[c]:
                    if i != k and j != k:
                        wcnf.append(  # if tref(i,k) and ref(k,j) -> tref(i,j)
                            [
                                -lm.getid(BiDirLiteral.tref, i, k),
                                -lm.getid(BiDirLiteral.ref, k, j),
                                lm.getid(BiDirLiteral.tref, i, j),
                            ]
                        )
        # acyclicity of tref: If tref(i,j) -> not tref(j,i)
    for i in range(n):
        for j in occ_others(occ1, text, i):
            wcnf.append([-lm.getid(BiDirLiteral.tref, i, j), -lm.getid(BiDirLiteral.tref, j, i)])

    # a root must be a beginning of a phrase: root(i) -> pstart(i)
    # sum_j ref(i,j) = 0 => pstart(i)
    # [or ref_[i,j] , pstart (i)]
    for i in range(n):
        wcnf.append(
            [lm.getid(BiDirLiteral.ref, i, j) for j in occ_others(occ1, text, i)] + [lm.getid(BiDirLiteral.pstart, i)]
        )

    # if i = 0 or j = 0 or T[i-1] \neq T[j-1]: not (ref(i,j)) or pstart(i)
    for c in occ1.keys():
        for i in occ1[c]:
            for j in occ_others(occ1, text, i):
                if i == 0 or j == 0 or text[i - 1] != text[j - 1]:
                    wcnf.append([-lm.getid(BiDirLiteral.ref, i, j), lm.getid(BiDirLiteral.pstart, i)])
    # for i,j > 0, and T[i] = T[j], T[i-1] = T[j-1]
    # if (not ref(i-1,j-1)) and ref(i,j) => pstart(i)
    # <=> ref(i-1,j-1) or not ref(i,j) or pstart(i)
    for c in occ1.keys():
        for i in occ1[c]:
            for j in occ_others(occ1, text, i):
                if i > 0 and j > 0 and text[i - 1] == text[j - 1]:
                    wcnf.append(
                        [
                            lm.getid(BiDirLiteral.ref, i - 1, j - 1),
                            -lm.getid(BiDirLiteral.ref, i, j),
                            lm.getid(BiDirLiteral.pstart, i),
                        ]
                    )

    # the first position is always a beginning of a phrase
    wcnf.append([lm.getid(BiDirLiteral.pstart, 0)])

    # objective: minimizes the number of factors
    for i in range(n):
        wcnf.append([-lm.getid(BiDirLiteral.pstart, i)], weight=1)

    return lm, wcnf


def min_bms(text: bytes, exp: Optional[BiDirExp] = None, contain_list: List[int] = []) -> BiDirType:
    """
    Compute the smallest bidirectional macro scheme (BMS).
    """
    total_start = time.time()
    lm, wcnf = bms_WCNF(text)
    for lname in lm.nvar.keys():
        logger.info(f"# of [{lname}] literals  = {lm.nvar[lname]}")

    for i in contain_list:
        fbeg0 = lm.getid(BiDirLiteral.pstart, i)
        wcnf.append([fbeg0])

    if exp:
        exp.time_prep = time.time() - total_start

    # solver = RC2(wcnf, verbose=3)
    solver = RC2(wcnf)
    sol = solver.compute()

    assert sol is not None
    sold = get_sold(sol)
    show_sol(lm, sold, text)
    factors = sol2bms(lm, sold, text)

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


def get_sold(sol: List[int]) -> Dict[int, bool]:
    """
    Compute dictionary res[literal_id] = True or False.
    """
    sold = dict()
    for x in sol:
        sold[abs(x)] = x > 0
    return sold


def bms_enumerate(text: bytes) -> Iterator[BiDirType]:
    lm, wcnf = bms_WCNF(text)
    solset = set()
    overlap = 0
    with RC2(wcnf) as solver:
        for sol in solver.enumerate():
            factors = sol2bms(lm, get_sold(sol), text)
            key = tuple(factors)
            if key not in solset:
                solset.add(key)
                logger.info(f"overlap solution = {overlap}")
                yield factors
            else:
                overlap += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Minimum bidirectional macro scheme (BMS)")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument(
        "--contains",
        nargs="+",
        type=int,
        help="list of text positions that must be included in the string attractor, starting with index 0",
        default=[],
    )
    parser.add_argument(
        "--log_level",
        type=str,
        help="log level, DEBUG/INFO/CRITICAL",
        default="CRITICAL",
    )

    args = parser.parse_args()
    if (args.file == "" and args.str == "") or (args.log_level not in ["DEBUG", "INFO", "CRITICAL"]):
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
    exp.algo = "bms-sat-fast"
    exp.file_name = os.path.basename(args.file)
    exp.file_len = len(text)
    factors_sol = min_bms(text, exp, args.contains)
    exp.factors = factors_sol
    exp.factor_size = len(factors_sol)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
