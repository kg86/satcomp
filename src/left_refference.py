# 左参照をする圧縮スキーム
# text長nに対してO(n^4)のリテラルが作成される

import sys
import argparse
from logging import getLogger, DEBUG, INFO, StreamHandler
from typing import Dict, List

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


# logging.basicConfig(
#     filename="left_refference.py.log",
#     filemode="w",
#     encoding="utf-8",
#     level=logger.DEBUG,
# )

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2

from mysat import *
import lz77


class LRLiteralManaer(LiteralManager):
    def __init__(self):
        self.lit_fbeg = "factor_begin"
        # self.lit_lmc = "left_most_character"
        self.lit_cmatch = "character_match"
        self.lit_lref = "left_reference"
        self.lit_match_next = "match_next"
        self.validf = dict()
        super().__init__()

    def id(self, *opt) -> int:
        res = super().newid(*opt)
        # assertion
        if opt[0] == self.lit_cmatch:
            assert opt[1] > opt[2]
        elif opt[0] == self.lit_lref:
            assert opt[1] >= opt[2]
        elif opt[0] == self.lit_match_next:
            assert opt[1] > opt[2]
        if opt[0] in self.validf:
            self.validf[opt[0]](opt)
        return res


def left_most_characters(text: bytes) -> Dict[int, int]:
    res = dict()
    for i, c in enumerate(text):
        if c not in res:
            res[c] = i
    return res


def occ_chars(text: bytes) -> Dict[int, List[int]]:
    res = dict()
    for i, c in enumerate(text):
        if c not in res:
            res[c] = []
        res[c].append(i)
    return res


def lit_by_id(sol: List[int], id: int):
    """returns boolean assignment of id"""
    return sol[id - 1] > 0


def get_lref(lm: LRLiteralManaer, sol: List[int], i: int):
    res = []
    for j in range(i + 1):
        if lm.contains(lm.lit_lref, i, j):
            # logger.info("contain", lm.contain(lm.lit_lref, i, j))
            # logger.info(f"lref({i}, {j})", lm.id(lm.lit_lref, i, j))
            lref = lit_by_id(sol, lm.id(lm.lit_lref, i, j))
            res.append((i, j, lref))
    return res


def sol2factors(lm: LRLiteralManaer, sol: List[int], text_len):
    """
    returns a list contains beginning positions of factors.
    The result also contains the position 0 and text length.
    """
    res = []
    for i in range(text_len):
        fbeg = lit_by_id(sol, lm.id(lm.lit_fbeg, i))
        if fbeg:
            res.append(i)
            # logger.info("lref", get_lref(lm, sol, i))
    res.append(text_len)
    return res


def sol2lz77(lm: LRLiteralManaer, sol: List[int], text: bytes) -> lz77.LZType:
    """returns LZ77"""
    res = lz77.LZType([])
    factors = sol2factors(lm, sol, len(text))
    for i, fbeg in enumerate(factors[:-1]):
        self_ref = lit_by_id(sol, lm.id(lm.lit_lref, fbeg, fbeg))
        if self_ref:
            res.append((-1, text[fbeg]))
        else:
            for j in range(fbeg):
                if text[fbeg] != text[j]:
                    continue
                ref = lit_by_id(sol, lm.id(lm.lit_lref, fbeg, j))
                if ref:
                    res.append((j, factors[i + 1] - fbeg))
                    break
    return res


def sol2lz772(lm: LRLiteralManaer, sol: List[int], text: bytes) -> lz77.LZType:
    """returns LZ77"""
    res = lz77.LZType([])
    factors = sol2factors(lm, sol, len(text))
    logger.info("factors", factors)
    pinfo = defaultdict(list)
    for i in range(len(text)):
        # if lit_by_id(sol, lm.id(lm.lit_fbeg, i)):
        #     pinfo[i].append(f"fbeg {chr(text[i])}")
        for j in range(i):
            if text[i] != text[j] or i + 1 == len(text):
                continue
            if lit_by_id(sol, lm.id(lm.lit_lref, i, j)):
                pinfo[i].append(f"refer text[{j}]={chr(text[j])}")

            nm = lit_by_id(sol, lm.id(lm.lit_match_next, i, j))
            if nm:
                assert text[i] == text[j]
                pinfo[i].append(f"nm ({i}, {j}, {chr(text[j+1])})")
    # for i in sorted(pinfo.keys()):
    #     logger.info(i, pinfo[i])

    for i, fbeg in enumerate(factors[:-1]):
        self_ref = lit_by_id(sol, lm.id(lm.lit_lref, fbeg, fbeg))
        if self_ref:
            res.append((-1, text[fbeg]))
        else:
            found = False
            for j in range(fbeg):
                if text[fbeg] != text[j]:
                    continue
                # logger.info(len(sol), fbeg, lm.id(lm.lit_lref, fbeg, j))
                ref = lit_by_id(sol, lm.id(lm.lit_lref, fbeg, j))
                if ref:
                    found = True
                    res.append((j, factors[i + 1] - fbeg))
                    break
            if not found:
                logger.info(lit_by_id(sol, lm.id(lm.lit_lref, fbeg, fbeg)))
                logger.info("not found", fbeg)
            assert found
    return res


def sol2lz773(
    lm: LRLiteralManaer, sol: List[int], text: bytes, lmc, match_left
) -> lz77.LZType:
    """returns LZ77"""
    res = lz77.LZType([])
    factors = sol2factors(lm, sol, len(text))
    logger.info("factors", factors)
    pinfo = defaultdict(list)
    for i in range(len(text)):
        if i == lmc[text[i]]:
            pinfo[i].append(f"lmc [{chr(text[i])}]")
        for j in match_left[i]:
            if lit_by_id(sol, lm.id(lm.lit_lref, i, j)):
                pinfo[i].append(f"refer text[{j}]={chr(text[j])}")
        logger.debug(f"{i} {pinfo[i]}")

    for fi, i in enumerate(factors[:-1]):
        if i == lmc[text[i]]:
            res.append((-1, text[i]))
        else:
            found = False
            for j in match_left[i]:
                # logger.info(len(sol), fbeg, lm.id(lm.lit_lref, fbeg, j))
                ref = lit_by_id(sol, lm.id(lm.lit_lref, i, j))
                if ref:
                    found = True
                    res.append((j, factors[fi + 1] - i))
                    break
            if not found:
                # logger.info(lit_by_id(sol, lm.id(lm.lit_lref, fbeg, fbeg)))
                logger.error(f"{i} does not refer to any positions")
            assert found
    return res


def assert_sol_continuous(sol: List[int]) -> None:
    for i in range(1, len(sol)):
        assert abs(sol[i]) - abs(sol[i - 1])


def min_lref_scheme(text: bytes):
    n = len(text)
    wcnf = WCNF()
    # symclauses = []
    lm = LRLiteralManaer()
    wcnf.append([lm.sym2id(lm.true)])
    wcnf.append([-lm.sym2id(lm.false)])

    lmc = left_most_characters(text)

    # # of factorization
    logger.info(
        "# of factorization",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(n):
        # if fbi is True, a factor begins at i.
        fbeg = lm.id(lm.lit_fbeg, i)
        assert lm.contains(lm.lit_fbeg, i)
        # print(f"debug factor_beg of {i} is {fbi}")
        self_ref = lm.id(lm.lit_lref, i, i)
        # minimize the number of factors
        wcnf.append([-fbeg], weight=1)
        wcnf.append(pysat_if(self_ref, fbeg))
    # left most occurrence
    logger.info(
        "left most occurrence",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(n):
        self_ref = lm.id(lm.lit_lref, i, i)
        assert text[i] in lmc
        if i == lmc[text[i]]:
            # text[i] is the left most occurrence, so it refer to itself.
            wcnf.append([self_ref])
            for j in range(i):
                wcnf.append([-lm.id(lm.lit_lref, i, j)])
            # a factor of length 1 begins at i
            wcnf.append([lm.id(lm.lit_fbeg, i)])
            # the next factor begins at i+1
            if i + 1 < n:
                wcnf.append([lm.id(lm.lit_fbeg, i + 1)])
        else:
            wcnf.append([-self_ref])
    # match characters
    logger.info(
        "match characters",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(n):
        for j in range(i):
            cmij = lm.id(lm.lit_cmatch, i, j)
            if text[i] == text[j]:
                wcnf.append([cmij])
            else:
                wcnf.append([-cmij])
            # if i refer to j, text[i] must equal text[j]
            lref_i = lm.id(lm.lit_lref, i, j)
            matchij = lm.id(lm.lit_cmatch, i, j)
            wcnf.append(pysat_if(lref_i, matchij))
    # left reference
    logger.info(
        "left reference",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(1, n):
        if i % 50 == 0:
            print("i", i)
        for j in range(i - 1):
            # fbi = lm.sym(lm.lit_fbeg, i)
            # refpre = lm.sym(lm.lit_lref, i - 1, j)
            # refi = lm.sym(lm.lit_lref, i, j + 1)
            # # if i is not the beginning factor, and i-1 refer to j, i must refer to j+1.
            # symclauses.append(sympy_if(~fbi & refpre, refi))

            fbeg = lm.id(lm.lit_fbeg, i)
            lref_pre = lm.id(lm.lit_lref, i - 1, j)
            lref_i = lm.id(lm.lit_lref, i, j + 1)
            # if i is not the beginning factor, and i-1 refer to j, i must refer to j+1.
            # if (-fbi and refpre), then refi. it equals (fbi or -refpre or refi)
            wcnf.append([fbeg, -lref_pre, lref_i])
    # each position refers to previous position or itself.
    logger.info(
        "refer previous position or itself",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(n):
        if i % 50 == 0:
            print("i", i)
        refs = [lm.id(lm.lit_lref, i, j) for j in range(i + 1)]
        wcnf.extend(
            CardEnc.equals(
                refs, bound=1, encoding=EncType.pairwise, top_id=lm.top() + 1
            )
        )
        # wcnf.extend(
        #     CardEnc.equals(
        #         refs,
        #         bound=1,
        #         # encoding=EncType.totalizer,
        #         encoding=EncType.mtotalizer,
        #         top_id=lm.top() + 1,
        #     )
        # )

    # conver the sympy clauses to pysat clauses
    logger.info(
        "convert the sympy clauses to pysat clauses",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    # for sym_clause in symclauses:
    #     clauses = sympy_cnf_pysat(lm.new_id, sym_clause)
    #     wcnf.extend(clauses)

    print(
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}"
    )
    print("solver is running")
    solver = RC2(wcnf)
    sol = solver.compute()
    # print(sol)
    assert sol is not None
    return sol2lz77(lm, sol, text)
    # return sol2factors(lm, sol, n)


def min_lref_scheme2(text: bytes):
    n = len(text)
    wcnf = WCNF()
    # symclauses = []
    lm = LRLiteralManaer()
    wcnf.append([lm.sym2id(lm.true)])
    wcnf.append([-lm.sym2id(lm.false)])

    lmc = left_most_characters(text)

    match_left = defaultdict(list)
    for i in range(n):
        for j in range(i):
            if text[i] != text[j]:
                continue
            match_left[i].append(j)

    def verify_lref(opt):
        assert 0 <= opt[1] < n
        assert opt[1] == opt[2] or opt[2] in match_left[opt[1]]

    lm.validf[lm.lit_lref] = verify_lref

    # # of factorization
    logger.info(
        "# of factorization"
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}"
    )
    for i in range(n):
        # if fbi is True, a factor begins at i.
        fbeg = lm.id(lm.lit_fbeg, i)
        self_ref = lm.id(lm.lit_lref, i, i)
        # minimize the number of factors
        wcnf.append([-fbeg], weight=1)
        wcnf.append(pysat_if(self_ref, fbeg))
        # if fbi is False, i does not refer to any positions.
        for j in match_left[i] + [i]:
            ref = lm.id(lm.lit_lref, i, j)
            wcnf.append(pysat_if(-fbeg, -ref))
    # left most occurrence
    logger.info(
        "left most occurrence"
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}"
    )
    for i in range(n):
        self_ref = lm.id(lm.lit_lref, i, i)
        if i == lmc[text[i]]:
            # text[i] is the left most occurrence, so it refer to itself.
            wcnf.append([self_ref])
            # a factor of length 1 begins at i
            wcnf.append([lm.id(lm.lit_fbeg, i)])
            # the next factor begins at i+1
            if i + 1 < n:
                wcnf.append([lm.id(lm.lit_fbeg, i + 1)])
        else:
            wcnf.append([-self_ref])
    # next match
    for i in range(n - 1):
        for j in match_left[i]:
            nm = lm.id(lm.lit_match_next, i, j)
            if text[i + 1] != text[j + 1]:
                wcnf.append([-nm])
    # match characters

    # left reference
    print(
        "left reference",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(n - 1):
        if i % 50 == 0:
            print("i", i)
        for j in match_left[i]:
            fbeg0 = lm.id(lm.lit_fbeg, i)
            fbeg1 = lm.id(lm.lit_fbeg, i + 1)
            fbeg2 = lm.id(lm.lit_fbeg, i + 2)
            nm0 = lm.id(lm.lit_match_next, i, j)
            nm1 = lm.id(lm.lit_match_next, i + 1, j + 1)
            lref = lm.id(lm.lit_lref, i, j)
            # fbeg1 must refer to left
            wcnf.append(pysat_if(fbeg1, -nm0))
            wcnf.append(pysat_if_and_then_or([fbeg0, -fbeg1, lref], [nm0]))
            wcnf.append(pysat_if_and_then_or([nm0, -fbeg1, -fbeg2], [nm1]))

    # each position refers to previous position or itself.
    print(
        "refer previous position or itself",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(n):
        refs = [lm.id(lm.lit_lref, i, j) for j in match_left[i] + [i]]
        nm1s = []
        if i > 1:
            nm1s = [lm.id(lm.lit_match_next, i - 1, j) for j in match_left[i - 1]]
        if i % 50 == 0:
            print(
                f"i={i} text[i-1:i+1]={text[max(0, i-1):min(len(text), i+1)]} refs={len(refs)}, nm1s={len(nm1s)}"
            )
        wcnf.extend(
            CardEnc.equals(
                refs + nm1s, bound=1, encoding=EncType.pairwise, top_id=lm.top() + 1
            )
        )

    # conver the sympy clauses to pysat clauses
    print(
        "convert the sympy clauses to pysat clauses",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )

    print(
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}"
    )
    print("solver is running")
    solver = RC2(wcnf)
    sol = solver.compute()
    # print(sol)
    assert sol is not None
    assert_sol_continuous(sol)
    return sol2lz772(lm, sol, text)
    # return sol2factors(lm, sol, n)


def min_lref_scheme3(text: bytes):
    n = len(text)
    wcnf = WCNF()
    # symclauses = []
    lm = LRLiteralManaer()
    wcnf.append([lm.sym2id(lm.true)])
    wcnf.append([-lm.sym2id(lm.false)])

    lmc = left_most_characters(text)

    match_left = defaultdict(list)
    for i in range(n):
        for j in range(i):
            if text[i] != text[j]:
                continue
            match_left[i].append(j)

    def verify_lref(opt):
        assert 0 <= opt[1] < n
        assert opt[1] == opt[2] or opt[2] in match_left[opt[1]]

    lm.validf[lm.lit_lref] = verify_lref

    logger.info(
        "# of factorization"
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}"
    )
    # objective
    for i in range(n):
        # if fbi is True, a factor begins at i.
        fbeg = lm.id(lm.lit_fbeg, i)
        # minimize the number of factors
        wcnf.append([-fbeg], weight=1)
    # left most occurrence
    logger.info(
        "left most occurrence"
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}"
    )
    for i in lmc.values():
        wcnf.append([lm.id(lm.lit_fbeg, i)])
        if i + 1 < n:
            wcnf.append([lm.id(lm.lit_fbeg, i + 1)])
    # next match
    for i in range(n - 1):
        if i == lmc[text[i]]:
            continue
        for j in match_left[i]:
            lref0 = lm.id(lm.lit_lref, i, j)
            if j + 1 in match_left[i + 1]:
                lref1 = lm.id(lm.lit_lref, i + 1, j + 1)
                wcnf.append(pysat_if(lref0, lref1))
            else:
                fbeg1 = lm.id(lm.lit_fbeg, i + 1)
                wcnf.append(pysat_if(lref0, fbeg1))

    # each position refers to previous position or itself.
    print(
        "refer previous position or itself",
        f"# of nv = {wcnf.nv}, # of hard clauses = {len(wcnf.hard)}, # of soft clauses = {len(wcnf.soft)}",
    )
    for i in range(n):
        if i == lmc[text[i]]:
            continue
        refs = [lm.id(lm.lit_lref, i, j) for j in match_left[i]]
        if i % 100 == 0:
            print(f"i={i}")
        wcnf.extend(
            CardEnc.equals(
                refs, bound=1, encoding=EncType.pairwise, top_id=lm.top() + 1
            )
        )

    print("solver is running")
    solver = RC2(wcnf)
    sol = solver.compute()
    # print(sol)
    assert sol is not None
    assert_sol_continuous(sol)
    return sol2lz773(lm, sol, text, lmc, match_left)
    # return sol2factors(lm, sol, n)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute LZ77")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--output", type=str, help="output file", default="")

    args = parser.parse_args()
    if args.file == "":
        parser.print_help()
        sys.exit()
    return args


if __name__ == "__main__":
    args = parse_args()

    text = open(args.file, "rb").read()
    # lz77_solver = min_lref_scheme(text)
    # lz77_solver = min_lref_scheme2(text)
    lz77_solver = min_lref_scheme3(text)
    # n = 0
    # for i, f in enumerate(lz77_solver):
    #     if f[0] == -1:
    #         logger.info(n, f[0], chr(f[1]))
    #         n += 1
    #     else:
    #         logger.info(n, f, text[f[0] : f[0] + f[1]].decode("utf8"))
    #         n += f[1]
    logger.info(f"lz77 {lz77_solver}")
    fs = lz77.factor_strs(lz77_solver)
    for f in fs:
        sys.stdout.write(f.decode("utf8") + " ")
    # logger.info(lz77_solver)
    lz77_true = lz77.encode(text)
    logger.info(f"# of lz77 = {len(lz77_true)}, # of lz77 solver = {len(lz77_solver)}")
    logger.info(f"is this lz77? {str(lz77.equal(text, lz77_true, lz77_solver))}")
    logger.info(f"can decode? {text == lz77.decode(lz77_solver)}")
    factors = [f.decode("utf8") for f in lz77.factor_strs(lz77_solver)]
    # logger.info(factors)
