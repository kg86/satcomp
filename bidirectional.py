# bidirectional schemeを計算する

from collections import namedtuple
import sys
import argparse
import os
import subprocess
from logging import getLogger, DEBUG, INFO, StreamHandler, Formatter

from left_refference import LRLiteralManaer
from mytimer import Timer
import lz77

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
FORMAT = "[%(lineno)s - %(funcName)10s() ] %(message)s"
formatter = Formatter(FORMAT)
handler.setFormatter(formatter)
logger.setLevel(DEBUG)
# logger.setLevel(INFO)
logger.addHandler(handler)


from dataclasses import dataclass

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2

from mysat import *


@dataclass
class Literal:
    link_to: str = "link_to"
    can_link: str = "can_link"
    root: str = "root"
    fbeg: str = "factor_begin"
    ref: str = "ref"


class BiDirLiteralManager(LiteralManager):
    def __init__(self, text: bytes, max_depth: int):
        self.lit = Literal()
        self.text = text
        self.max_depth = max_depth
        self.n = len(self.text)
        self.validf = {
            self.lit.link_to: self.valid_link,
            # self.lit.can_link: self.valid_can_link,
            self.lit.root: self.valid_root,
            self.lit.ref: self.valid_ref,
        }
        super().__init__()

    def id(self, *opt) -> int:
        res = super().id(*opt)
        if opt[0] in self.validf:
            self.validf[opt[0]](opt)
        return res

    def valid_link(self, opt):
        assert len(opt) == 4
        assert opt[0] == self.lit.link_to
        # depth
        assert 0 <= opt[1] < self.max_depth - 1
        # link opt[2] to opt[3]
        assert 0 <= opt[2], opt[3] < self.n
        assert opt[2] != opt[3]
        assert self.text[opt[2]] == self.text[opt[3]]

    # def valid_can_link(self, opt):
    #     assert len(opt) == 3
    #     assert 0 <= opt[1] < self.n - 1
    #     assert 0 <= opt[2] < self.n
    #     assert self.text[opt[2]] == self.text[opt[3]]

    def valid_root(self, opt):
        # opt = (name, depth, pos)
        # assert len(opt) == 3
        # assert opt[1] < self.max_depth
        # assert 0 <= opt[2] < self.n
        assert len(opt) == 2
        assert 0 <= opt[1] < self.n

    def valid_ref(self, opt):
        assert len(opt) == 3
        assert opt[1] != opt[2]
        assert 0 <= opt[1], opt[2] < self.n
        assert self.text[opt[1]] == self.text[opt[2]]


def pairs(n: int):
    for i in range(n):
        for j in range(n):
            if i != j:
                yield (i, j)


def lit_by_id(sol: list[int], id: int):
    return sol[id - 1] > 0


def sol2links(lm: BiDirLiteralManager, sol: list[int], text: bytes):
    n = len(text)
    occ = make_occa(text)
    refs = dict()
    for i in range(n):
        if lit_by_id(sol, lm.id(lm.lit.root, 0, i)):
            refs[i] = -1
    for depth in range(n - 1):
        for i in range(n):
            for j in occ[text[i]]:
                if i == j:
                    continue
                key = (lm.lit.link_to, depth, i, j)

                if lm.contains(*key) and lit_by_id(sol, lm.getid(*key)):
                    refs[j] = i
    logger.debug(refs)
    return refs


def sol2links2(lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes):
    n = len(text)
    occ = make_occa(text)
    refs = dict()
    for i in range(n):
        if sol[lm.id(lm.lit.root, i)]:
            refs[i] = -1
    for depth in range(n - 1):
        for i in range(n):
            for j in occ[text[i]]:
                if i == j:
                    continue
                key = (lm.lit.link_to, depth, i, j)

                if lm.contains(*key) and sol[lm.getid(*key)]:
                    refs[j] = i
    logger.debug(f"refs={refs}")
    return refs


# def show_sol(lm: BiDirLiteralManager, sol: list[int], text: bytes):
#     n = len(text)
#     occ = make_occa(text)
#     pinfo = defaultdict(list)
#     refs = sol2links(lm, sol, text)
#     for i in range(n):
#         pinfo[i].append(chr(text[i]))
#         pinfo[i].append(f"(ref, {refs[i]})")
#         key = (lm.lit.root, 0, i)
#         lid = lm.id(*key)
#         if lit_by_id(sol, lid):
#             pinfo[i].append(str(key))
#         fbeg_key = (lm.lit.fbeg, i)
#         # logger.debug(f"{key}, {lm.id(*key)}, {lit_by_id(sol, lm.id(*key))}")
#         if lit_by_id(sol, lm.id(*fbeg_key)):
#             pinfo[i].append(str(fbeg_key))
#     for depth in range(1):
#         for i in range(n):
#             for j in occ[text[i]]:
#                 if i == j:
#                     continue
#                 link_key = (lm.lit.link_to, depth, i, j)
#                 if lit_by_id(sol, lm.id(*link_key)):
#                     pinfo[i].append(str(link_key))
#     for i in range(n):
#         logger.debug(f"i={i} " + ", ".join(pinfo[i]))

#     # for step in range(n - 1):
#     #     link = None
#     #     for (i, j) in pairs(n):
#     #         lid = lm.id(lm.lit.link_to, step, i, j)
#     #         lit = lit_by_id(sol, lid)
#     #         if lit:
#     #             logger.debug(lm.id2str(lid))


def show_sol2(lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes):
    n = len(text)
    occ = make_occa(text)
    pinfo = defaultdict(list)
    refs = sol2links2(lm, sol, text)

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
        # if sol[lm.getid(lm.lit.root, i)]:
        #     pinfo[i].append(f"(root, {i})")
        for j in occ[text[i]]:
            if i == j:
                continue
            key = (lm.lit.ref, i, j)
            if sol[lm.getid(*key)]:
                pinfo[i].append(str(key))
        # pinfo[i].append(f"(ref, {refs[i]})")
        key = (lm.lit.root, i)
        lid = lm.id(*key)
        if sol[lid]:
            pinfo[i].append(str(key))
        fbeg_key = (lm.lit.fbeg, i)
        # logger.debug(f"{key}, {lm.id(*key)}, {lit_by_id(sol, lm.id(*key))}")
        if sol[lm.id(*fbeg_key)]:
            pinfo[i].append(str(fbeg_key))
        key_link = (lm.lit.link_to,)
        for key in link_to(i):
            pinfo[i].append(f"{key}")
    # for depth in range(1):
    #     for i in range(n):
    #         for j in occ[text[i]]:
    #             if i == j:
    #                 continue
    #             link_key = (lm.lit.link_to, depth, i, j)
    #             if sol[lm.id(*link_key)]:
    #                 pinfo[i].append(str(link_key))
    for i in range(n):
        logger.debug(f"i={i} " + ", ".join(pinfo[i]))

    # for step in range(n - 1):
    #     link = None
    #     for (i, j) in pairs(n):
    #         lid = lm.id(lm.lit.link_to, step, i, j)
    #         lit = lit_by_id(sol, lid)
    #         if lit:
    #             logger.debug(lm.id2str(lid))


def pysat_equal(lm: BiDirLiteralManager, bound: int, lits: list[int]):
    return CardEnc.equals(lits, bound=bound, encoding=EncType.pairwise, vpool=lm.vpool)


def make_occa(text: bytes) -> dict[int, list[int]]:
    occ = defaultdict(list)
    for i in range(len(text)):
        occ[text[i]].append(i)
    return occ


from typing import NewType

BiDirType = NewType("BiDirType", list[tuple[int, int]])


def decode_len(factors: BiDirType) -> int:
    res = 0
    for f in factors:
        res += 1 if f[0] == -1 else f[1]
    return res


def decode(factors: BiDirType) -> bytes:
    n = decode_len(factors)
    # none = 'x'
    res = [-1 for _ in range(n)]
    i = 0
    fs = [list(f) for f in factors]
    fbegs = []
    for f in fs:
        fbegs.append(i)
        if f[0] == -1:
            res[i] = f[1]
            i += 1
        else:
            i += f[1]

    logger.debug(f"fbegs={fbegs}")
    for step in range(n):
        found = sum(1 for i in res if i != -1)
        logger.debug(f"step={step}: found={found}/{n}")
        if found == n:
            break
        for fi, f in enumerate(fs):
            refi, reflen = f
            if refi == -1:
                continue
            i = fbegs[fi]
            count = 0
            for j in range(reflen):
                if res[refi + j] != -1:
                    count += 1
                    res[i + j] = res[refi + j]
            if reflen == count:
                fs[fi][0] = -1

    logger.debug(f"decode={res}")
    return bytes(res)


def sol2fbegs(lm: BiDirLiteralManager, sol: list[int], text: bytes) -> list[int]:
    fbegs = []
    n = len(text)
    for i in range(n):
        if lit_by_id(sol, lm.id(lm.lit.fbeg, i)):
            fbegs.append(i)
    fbegs.append(n)
    return fbegs


def sol2lits(lm: BiDirLiteralManager, sol: list[int], lit_name: str) -> list:
    res = []
    for id, obj in lm.vpool.id2obj.items():
        assert isinstance(id, int)
        if obj[0] == lit_name:
            res.append((obj, lit_by_id(sol, id)))

    return res


def sol2lits2(lm: BiDirLiteralManager, sol: dict[int, bool], lit_name: str) -> list:
    logger.debug(f"show lits {lit_name}")
    res = []
    for id, obj in lm.vpool.id2obj.items():
        assert isinstance(id, int)
        if obj[0] == lit_name:
            res.append((obj, sol[id]))

    return res


def sol2bidirectional(
    lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes
) -> BiDirType:
    res = BiDirType([])
    fbegs = []
    occ = make_occa(text)
    n = len(text)
    refs = sol2links2(lm, sol, text)
    # for i, j in sorted(refs.items()):
    #     logger.debug(f"(ref, {i}, {j})")
    for i in range(n):
        if sol[lm.id(lm.lit.fbeg, i)]:
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


# def bidirectional(text: bytes):
#     n = len(text)
#     wcnf = WCNF()
#     lm = BiDirLiteralManager(text)
#     occ = make_occa(text)

#     # def valid_link(opt):
#     #     assert len(opt) == 4
#     #     assert 0 <= opt[1] < n - 1
#     #     assert 0 <= opt[2], opt[3] < n
#     #     assert opt[2] != opt[3]

#     # def valid_can_link(opt):
#     #     assert len(opt) == 3
#     #     assert 0 <= opt[1] < n - 1
#     #     assert 0 <= opt[2] < n

#     # lm.validf[lm.lit.link_to] = valid_link
#     # lm.validf[lm.lit.can_link] = valid_can_link
#     wcnf.append([lm.sym2id(lm.true)])
#     wcnf.append([lm.sym2id(lm.false)])

#     # 位置0はfactorの開始位置となる
#     wcnf.append([lm.id(lm.lit.fbeg, 0)])
#     # rootとrootの次の位置はfactorの開始位置
#     for i in range(n):
#         root0 = lm.id(lm.lit.root, 0, i)
#         fbeg0 = lm.id(lm.lit.fbeg, i)
#         wcnf.append([-fbeg0], weight=1)
#         wcnf.append(pysat_if(root0, fbeg0))
#         if i + 1 < n:
#             fbeg1 = lm.id(lm.lit.fbeg, i + 1)
#             wcnf.append(pysat_if(root0, fbeg1))

#     # 2文字一致する位置リスト
#     match2 = defaultdict(list)
#     for i in range(n - 1):
#         for j in range(i):
#             if text[i : i + 2] == text[j : j + 2]:
#                 match2[i].append(j)
#                 match2[j].append(i)
#     # i to jへのリンクが有り、i+1 to j+1へリンクが可能であれば、必ずリンクする
#     for depth in range(n - 1):
#         for i in range(n - 1):
#             for j in match2[i]:
#                 link0 = lm.id(lm.lit.link_to, depth, i, j)
#                 link1 = lm.id(lm.lit.link_to, depth, i + 1, j + 1)
#                 rooti1 = lm.id(lm.lit.root, depth, i + 1)
#                 rootj1 = lm.id(lm.lit.root, depth, j + 1)
#                 if_body = [link0, -rooti1, rootj1]
#                 if_then = link1
#                 wcnf.append(pysat_if_all(if_body, if_then))

#     # factorの開始位置
#     # i to jへのリンクが有り T[i-1]!=T[j-1]であれば、位置jはfactorの開始位置
#     # i to jへのリンクが有り T[i-1]==T[j-1]かつi-1 to j-1のリンクがなければ、位置jはfactorの開始位置
#     # i to jへのリンクが有り i=0なら、位置jはfactorの開始位置
#     for depth in range(n - 1):
#         for i in range(n):
#             for j in occ[text[i]]:
#                 if i == j or j == 0:
#                     continue
#                 link_ij0 = lm.id(lm.lit.link_to, depth, i, j)
#                 fbeg_j = lm.id(lm.lit.fbeg, j)
#                 if i == 0 or text[i - 1] != text[j - 1]:
#                     # logger.debug(f"(depth, i, j)={(depth, i,j)}")
#                     if_body = link_ij0
#                     if_then = fbeg_j
#                     wcnf.append(pysat_if(if_body, if_then))
#                 else:
#                     link_ij1 = lm.id(lm.lit.link_to, depth, i - 1, j - 1)
#                     if_body = [-link_ij1, link_ij0]
#                     if_then = fbeg_j
#                     wcnf.append(pysat_if_all(if_body, if_then))
#                     # else:
#                     #     if_body = link_ij0
#                     #     if_then = fbeg_j
#                     #     wcnf.append(pysat_if(if_body, if_then))

#     # 各文字について、rootが１つ以上必ず存在する
#     # for pos in occ.values():
#     #     roots = [lm.id(lm.lit.root, 0, i) for i in pos]
#     #     wcnf.append(pysat_atleast_one(roots))

#     # root propagation
#     # 深さdでrootなら深さd+1でもroot
#     for depth in range(n - 1):
#         for i in range(n):
#             root0 = lm.id(lm.lit.root, depth, i)
#             root1 = lm.id(lm.lit.root, depth + 1, i)
#             wcnf.append(pysat_if(root0, root1))
#     # 深さn-1ではすべての位置がrootとなる
#     for i in range(n):
#         root = lm.id(lm.lit.root, n - 1, i)
#         wcnf.append([root], weight=n + 1)

#     # rootでないノードからlinkは出ない
#     for depth in range(n - 1):
#         for i in range(n):
#             root0 = lm.id(lm.lit.root, depth, i)
#             for j in occ[text[i]]:
#                 if i == j:
#                     continue
#                 link_ij = lm.id(lm.lit.link_to, depth, i, j)
#                 wcnf.append(pysat_if(-root0, -link_ij))

#     # i to j at depth>1のリンクが有るならば, k to j at depth-1 or
#     for depth in range(1, n - 1):
#         for i in range(n):
#             for j in occ[text[i]]:
#                 if i == j:
#                     continue
#                 link_ij = lm.sym(lm.lit.link_to, depth, i, j)
#                 links_to_i = [
#                     lm.sym(lm.lit.link_to, depth - 1, k, i)
#                     for k in occ[text[i]]
#                     if k != i and k != j
#                 ]
#                 eq = sympy_if(link_ij, atleast_one(links_to_i))
#                 wcnf.extend(sympy_cnf_pysat(lm.new_id, eq))

#     # 各位置は参照を持つ
#     ## rootから
#     ## if link i to j at depth, i at depth+1 is root
#     for depth in range(n - 1):
#         for i in range(n):
#             root1 = lm.getid(lm.lit.root, depth + 1, i)
#             for j in occ[text[i]]:
#                 if i == j:
#                     continue
#                 link0 = lm.id(lm.lit.link_to, depth, i, j)
#                 wcnf.append(pysat_if(link0, root1))

#     # for c, pos in occ.items():
#     #     print(c, pos)
#     #     for i in pos:
#     #         root = lm.id(lm.lit.root, 0, i)
#     #         for j in pos:
#     #             if i != j:
#     #                 link_to = lm.id(lm.lit.link_to, 0, i, j)
#     #                 wcnf.append(pysat_if(link_to, root))

#     # position jへのリンクの数は高々１
#     # iへのリンクの数+depth=0でのrootは必ず１
#     for i in range(n):
#         rooti = lm.getid(lm.lit.root, 0, i)
#         links_i = [
#             lm.id(lm.lit.link_to, depth, j, i)
#             for depth in range(n - 1)
#             for j in occ[text[i]]
#             if i != j
#         ]
#         wcnf.extend(pysat_equal(lm, 1, links_i + [rooti]))
#         # wcnf.extend(
#         #     CardEnc.equals(
#         #         links_i + [rooti],
#         #         bound=1,
#         #         # encoding=EncType.totalizer,
#         #         encoding=EncType.mtotalizer,
#         #         top_id=lm.top() + 1,
#         #     )
#         # )

#     # 各位置はrootもしくは参照される
#     # for i in range(n):
#     #     root = lm.id(lm.lit.root, 0, i)
#     #     links = [lm.id(lm.lit.link_to, 0, j, i) for j in occ[text[i]] if i != j]
#     #     wcnf.extend(pysat_equal(lm, 1, links + [root]))

#     # solver runs

#     logger.info(
#         f"#literals = {lm.top()}, # hard clauses={len(wcnf.hard)}, # of soft clauses={len(wcnf.soft)}"
#     )
#     solver = RC2(wcnf)
#     sol = solver.compute()
#     assert sol is not None
#     sold = dict()
#     for x in sol:
#         sold[abs(x)] = x > 0
#     # logger.debug(sol)
#     def show_lits(lits):
#         for lit in sorted(lits):
#             if lit[1]:
#                 logger.debug(lit[0])

#     show_lits(sol2lits2(lm, sold, lm.lit.link_to))
#     # show_lits(sol2lits2(lm, sold, lm.lit.root))
#     show_sol2(lm, sold, text)
#     factors = sol2bidirectional(lm, sold, text)

#     logger.debug(factors)
#     logger.debug(f"original={text}")
#     logger.debug(f"decode={decode(factors).decode('utf8')}")
#     assert decode(factors) == text
#     return factors


def make_occa2(text: bytes) -> dict[int, list[int]]:
    match2 = defaultdict(list)
    for i in range(len(text) - 1):
        for j in range(i):
            if text[i : i + 2] == text[j : j + 2]:
                match2[i].append(j)
                match2[j].append(i)
    return match2


# def bidirectional2(text: bytes):
#     logger.info("bidirectional_solver start")
#     n = len(text)
#     wcnf = WCNF()
#     lm = BiDirLiteralManager(text)
#     occ1 = make_occa(text)
#     occ2 = make_occa2(text)
#     lz77fs = lz77.encode(text)

#     wcnf.append([lm.sym2id(lm.true)])
#     wcnf.append([lm.sym2id(lm.false)])

#     # literalを列挙する
#     # これ以降literalの新規作成は行わないようにする
#     lits = [lm.sym2id(lm.true)]
#     for depth in range(n - 1):
#         for i in range(n):
#             for j in occ1[text[i]]:
#                 if i == j:
#                     continue
#                 lits.append(lm.id(lm.lit.link_to, depth, i, j))
#     for depth in range(n):
#         for i in range(n):
#             lits.append(lm.id(lm.lit.root, depth, i))
#     for i in range(n):
#         lits.append(lm.id(lm.lit.fbeg, i))
#     for i in range(n):
#         for j in occ1[text[i]]:
#             if i == j:
#                 continue
#             lits.append(lm.id(lm.lit.ref, i, j))
#     wcnf.append(lits)

#     # rootとrootの次の位置はfactorの開始位置
#     for i in range(n):
#         root0 = lm.getid(lm.lit.root, 0, i)
#         fbeg0 = lm.getid(lm.lit.fbeg, i)
#         wcnf.append([-fbeg0], weight=1)
#         wcnf.append(pysat_if(root0, fbeg0))
#         if i + 1 < n:
#             fbeg1 = lm.getid(lm.lit.fbeg, i + 1)
#             wcnf.append(pysat_if(root0, fbeg1))

#     # i to jへのリンクが有り、i+1 to j+1へリンクが可能であれば、必ずリンクする
#     # for depth in range(n - 1):
#     #     for i in range(n - 1):
#     #         for j in occ2[i]:
#     #             link0 = lm.id(lm.lit.link_to, depth, i, j)
#     #             link1 = lm.id(lm.lit.link_to, depth, i + 1, j + 1)
#     #             rooti1 = lm.id(lm.lit.root, depth, i + 1)
#     #             rootj1 = lm.id(lm.lit.root, depth, j + 1)
#     #             if_body = [link0, -rooti1, rootj1]
#     #             if_then = link1
#     #             wcnf.append(pysat_if_all(if_body, if_then))

#     # factorの開始位置
#     # i to jへのリンクが有り T[i-1]!=T[j-1]であれば、位置jはfactorの開始位置
#     # i to jへのリンクが有り T[i-1]==T[j-1]かつi-1 to j-1のリンクがなければ、位置jはfactorの開始位置
#     # i to jへのリンクが有り i=0なら、位置jはfactorの開始位置
#     for depth in range(n - 1):
#         for i in range(n):
#             for j in occ1[text[i]]:
#                 if i == j or j == 0:
#                     continue
#                 link_ij0 = lm.getid(lm.lit.link_to, depth, i, j)
#                 fbeg_j = lm.getid(lm.lit.fbeg, j)
#                 # if i == 0 or text[i - 1] != text[j - 1]:
#                 #     # logger.debug(f"(depth, i, j)={(depth, i,j)}")
#                 #     if_body = link_ij0
#                 #     if_then = fbeg_j
#                 #     wcnf.append(pysat_if(if_body, if_then))
#                 # else:
#                 #     link_ij1 = lm.getid(lm.lit.link_to, depth, i - 1, j - 1)
#                 #     if_body = [-link_ij1, link_ij0]
#                 #     if_then = fbeg_j
#                 #     wcnf.append(pysat_if_all(if_body, if_then))
#     # ref
#     # linkからrefを決める
#     # refからfbegを決める
#     for i in range(n):
#         for j in occ1[text[i]]:
#             if i == j:
#                 continue
#             assert 0 <= i, j < n
#             ref0 = lm.getid(lm.lit.ref, j, i)
#             fbeg = lm.getid(lm.lit.fbeg, j)
#             for depth in range(n - 1):
#                 link = lm.getid(lm.lit.link_to, depth, i, j)
#                 wcnf.append(pysat_if(link, ref0))
#             # if i == 0 or j == 0:
#             #     # 位置0はfactorの開始位置
#             #     # 位置0を参照する位置はfactorの開始位置
#             #     wcnf.append(pysat_if(ref0, fbeg))
#             # if i > 0 and j > 0 and text[i - 1] != text[j - 1]:
#             if i == 0 or j == 0 or text[i - 1] != text[j - 1]:
#                 wcnf.append(pysat_if(ref0, fbeg))
#             if i > 0 and j > 0 and text[i - 1] == text[j - 1]:
#                 ref1 = lm.getid(lm.lit.ref, j - 1, i - 1)
#                 fbeg0 = lm.getid(lm.lit.fbeg, j)
#                 if i == 4 and j == 1:
#                     logger.debug(
#                         f"(i, j)={(i,j), {lm.id2str(ref1), lm.id2str(ref0), lm.id2str(fbeg0)}}"
#                     )
#                 wcnf.append(pysat_if_all([-ref1, ref0], fbeg0))
#                 wcnf.append(pysat_if_all([ref1, ref0], -fbeg0))
#     # 参照はたかだか１つ
#     for i in range(n):
#         refs = [lm.getid(lm.lit.ref, i, j) for j in occ1[text[i]] if i != j]
#         wcnf.extend(CardEnc.atmost(refs, bound=1, vpool=lm.vpool))

#     # 各文字について、rootが１つ以上必ず存在する
#     # for pos in occ.values():
#     #     roots = [lm.id(lm.lit.root, 0, i) for i in pos]
#     #     wcnf.append(pysat_atleast_one(roots))

#     # root propagation
#     # 深さdでrootなら深さd+1でもroot
#     for depth in range(n - 1):
#         for i in range(n):
#             root0 = lm.getid(lm.lit.root, depth, i)
#             root1 = lm.getid(lm.lit.root, depth + 1, i)
#             wcnf.append(pysat_if(root0, root1))
#             wcnf.append(pysat_if(-root1, -root0))
#     # 深さn-1ではすべての位置がrootとなる
#     # for i in range(n):
#     #     root = lm.getid(lm.lit.root, n - 1, i)
#     #     wcnf.append([root], weight=n + 1)

#     # rootでないノードからlinkは出ない
#     # for depth in range(n - 1):
#     #     for i in range(n):
#     #         root0 = lm.getid(lm.lit.root, depth, i)
#     #         for j in occ1[text[i]]:
#     #             if i == j:
#     #                 continue
#     #             link_ij = lm.getid(lm.lit.link_to, depth, i, j)
#     #             wcnf.append(pysat_if(-root0, -link_ij))

#     # i to j at depth>1のリンクが有るならば, k to j at depth-1 のリンクが必ず存在する
#     for depth in range(1, n - 1):
#         for i in range(n):
#             for j in occ1[text[i]]:
#                 if i == j:
#                     continue
#                 # link_ij = lm.sym(lm.lit.link_to, depth, i, j)
#                 link_ij = lm.id2sym(lm.getid(lm.lit.link_to, depth, i, j))
#                 links_to_i = [
#                     lm.id2sym(lm.getid(lm.lit.link_to, depth - 1, k, i))
#                     for k in occ1[text[i]]
#                     if k != i and k != j
#                 ]
#                 eq = sympy_if(link_ij, atleast_one(links_to_i))
#                 wcnf.extend(sympy_cnf_pysat(lm.new_id, eq))

#     # 各位置は参照を持つ
#     ## rootから
#     ## if link i to j at depth, i at depth+1 is root
#     for depth in range(n - 1):
#         for i in range(n):
#             root0 = lm.getid(lm.lit.root, depth, i)
#             for j in occ1[text[i]]:
#                 if i == j:
#                     continue
#                 root1 = lm.getid(lm.lit.root, depth + 1, j)
#                 ref0 = lm.getid(lm.lit.link_to, depth, i, j)
#                 wcnf.append(pysat_if(ref0, root1))
#                 wcnf.append(pysat_if(ref0, root0))

#     # for c, pos in occ.items():
#     #     print(c, pos)
#     #     for i in pos:
#     #         root = lm.id(lm.lit.root, 0, i)
#     #         for j in pos:
#     #             if i != j:
#     #                 link_to = lm.id(lm.lit.link_to, 0, i, j)
#     #                 wcnf.append(pysat_if(link_to, root))

#     # position jへのリンクの数は高々１
#     # iへのリンクの数+depth=0でのrootは必ず１
#     for i in range(n):
#         rooti = lm.getid(lm.lit.root, 0, i)
#         links_i = [
#             lm.id(lm.lit.link_to, depth, j, i)
#             for depth in range(n - 1)
#             for j in occ1[text[i]]
#             if i != j
#         ]
#         wcnf.extend(pysat_equal(lm, 1, links_i + [rooti]))

#     # 各位置はrootもしくは参照される
#     # for i in range(n):
#     #     root = lm.id(lm.lit.root, 0, i)
#     #     links = [lm.id(lm.lit.link_to, 0, j, i) for j in occ[text[i]] if i != j]
#     #     wcnf.extend(pysat_equal(lm, 1, links + [root]))

#     # solver runs

#     # assumptions
#     # opt_fbegs = [0, 5, 9, 12, 14, 15, 20]
#     # for i in range(n):
#     #     fbeg = lm.getid(lm.lit.fbeg, i)
#     #     if i in opt_fbegs:
#     #         wcnf.append([fbeg])
#     #     else:
#     #         wcnf.append([-fbeg])

#     logger.info(
#         f"#literals = {lm.top()}, # hard clauses={len(wcnf.hard)}, # of soft clauses={len(wcnf.soft)}"
#     )
#     solver = RC2(wcnf)
#     sol = solver.compute()
#     assert sol is not None
#     sold = dict()
#     for x in sol:
#         sold[abs(x)] = x > 0
#     # logger.debug(sol)
#     def show_lits(lits):
#         for lit in sorted(lits):
#             if lit[1]:
#                 logger.debug(lit[0])

#     show_lits(sol2lits2(lm, sold, lm.lit.link_to))
#     show_lits(sol2lits2(lm, sold, lm.lit.ref))
#     # show_lits(sol2lits2(lm, sold, lm.lit.root))
#     show_sol2(lm, sold, text)
#     factors = sol2bidirectional(lm, sold, text)

#     logger.debug(factors)
#     logger.debug(f"original={text}")
#     logger.debug(f"decode={decode(factors).decode('utf8')}")
#     assert decode(factors) == text
#     return factors


def bidirectional_v3(text: bytes):
    logger.info("bidirectional_solver start")
    n = len(text)
    wcnf = WCNF()
    lz77fs = lz77.encode(text)
    max_depth = min(len(lz77.encode(text)), len(lz77.encode(text[::-1]))) + 1
    lm = BiDirLiteralManager(text, max_depth)
    occ1 = make_occa(text)
    occ2 = make_occa2(text)
    logger.info(f"# of text = {len(text)}, # of lz77 = {len(lz77fs)}")

    wcnf.append([lm.sym2id(lm.true)])
    wcnf.append([lm.sym2id(lm.false)])

    # literalを列挙する
    # これ以降literalの新規作成は行わないようにする
    lits = [lm.sym2id(lm.true)]
    for depth in range(max_depth - 1):
        for i in range(n):
            for j in occ1[text[i]]:
                if i == j:
                    continue
                lits.append(lm.id(lm.lit.link_to, depth, i, j))
    for i in range(n):
        lits.append(lm.id(lm.lit.fbeg, i))
        lits.append(lm.id(lm.lit.root, i))
    for i in range(n):
        for j in occ1[text[i]]:
            if i == j:
                continue
            lits.append(lm.id(lm.lit.ref, i, j))
    wcnf.append(lits)

    # rootとrootの次の位置はfactorの開始位置
    for i in range(n):
        root0 = lm.getid(lm.lit.root, i)
        fbeg0 = lm.getid(lm.lit.fbeg, i)
        wcnf.append([-fbeg0], weight=1)
        wcnf.append(pysat_if(root0, fbeg0))
        if i + 1 < n:
            fbeg1 = lm.getid(lm.lit.fbeg, i + 1)
            wcnf.append(pysat_if(root0, fbeg1))

    # i to jへのリンクが有り、i+1 to j+1へリンクが可能であれば、必ずリンクする
    # for depth in range(n - 1):
    #     for i in range(n - 1):
    #         for j in occ2[i]:
    #             link0 = lm.id(lm.lit.link_to, depth, i, j)
    #             link1 = lm.id(lm.lit.link_to, depth, i + 1, j + 1)
    #             rooti1 = lm.id(lm.lit.root, depth, i + 1)
    #             rootj1 = lm.id(lm.lit.root, depth, j + 1)
    #             if_body = [link0, -rooti1, rootj1]
    #             if_then = link1
    #             wcnf.append(pysat_if_all(if_body, if_then))

    # factorの開始位置
    # i to jへのリンクが有り T[i-1]!=T[j-1]であれば、位置jはfactorの開始位置
    # i to jへのリンクが有り T[i-1]==T[j-1]かつi-1 to j-1のリンクがなければ、位置jはfactorの開始位置
    # i to jへのリンクが有り i=0なら、位置jはfactorの開始位置
    for depth in range(max_depth - 1):
        for i in range(n):
            for j in occ1[text[i]]:
                if i == j or j == 0:
                    continue
                link_ij0 = lm.getid(lm.lit.link_to, depth, i, j)
                fbeg_j = lm.getid(lm.lit.fbeg, j)
                # if i == 0 or text[i - 1] != text[j - 1]:
                #     # logger.debug(f"(depth, i, j)={(depth, i,j)}")
                #     if_body = link_ij0
                #     if_then = fbeg_j
                #     wcnf.append(pysat_if(if_body, if_then))
                # else:
                #     link_ij1 = lm.getid(lm.lit.link_to, depth, i - 1, j - 1)
                #     if_body = [-link_ij1, link_ij0]
                #     if_then = fbeg_j
                #     wcnf.append(pysat_if_all(if_body, if_then))
    # ref
    # linkからrefを決める
    # refからfbegを決める
    for i in range(n):
        for j in occ1[text[i]]:
            if i == j:
                continue
            assert 0 <= i, j < n
            ref_ji0 = lm.getid(lm.lit.ref, j, i)
            fbeg_j = lm.getid(lm.lit.fbeg, j)
            for depth in range(max_depth - 1):
                logger.debug(
                    f"if link {i} to {j} at depth {depth}, then ref {j} to {i}"
                )
                link_ij = lm.getid(lm.lit.link_to, depth, i, j)
                wcnf.append(pysat_if(link_ij, ref_ji0))
            # if i == 0 or j == 0:
            #     # 位置0はfactorの開始位置
            #     # 位置0を参照する位置はfactorの開始位置
            #     wcnf.append(pysat_if(ref0, fbeg))
            # if i > 0 and j > 0 and text[i - 1] != text[j - 1]:
            if i == 0 or j == 0 or text[i - 1] != text[j - 1]:
                logger.debug(f"if ref {j} to {i}, then fbeg {j}")
                wcnf.append(pysat_if(ref_ji0, fbeg_j))
            if i > 0 and j > 0 and text[i - 1] == text[j - 1]:
                ref_ji1 = lm.getid(lm.lit.ref, j - 1, i - 1)
                fbeg0 = lm.getid(lm.lit.fbeg, j)
                if i == 4 and j == 1:
                    logger.debug(
                        f"(i, j)={(i,j), {lm.id2str(ref_ji1), lm.id2str(ref_ji0), lm.id2str(fbeg0)}}"
                    )
                wcnf.append(pysat_if_all([-ref_ji1, ref_ji0], fbeg0))
                wcnf.append(pysat_if_all([ref_ji1, ref_ji0], -fbeg0))
    # 参照はたかだか１つ
    # rootなら参照はしない
    for i in range(n):
        refs = [lm.getid(lm.lit.ref, i, j) for j in occ1[text[i]] if i != j]
        wcnf.extend(CardEnc.atmost(refs, bound=1, vpool=lm.vpool))
        root_i = lm.getid(lm.lit.root, i)
        for j in occ1[text[i]]:
            if i == j:
                continue
            ref_ij = lm.getid(lm.lit.ref, i, j)
            link_ij = lm.getid(lm.lit.link_to, 0, i, j)
            wcnf.append(pysat_if(root_i, -ref_ij))
            wcnf.append(pysat_if(-root_i, -link_ij))

    # 各文字について、rootが１つ以上必ず存在する
    # for pos in occ.values():
    #     roots = [lm.id(lm.lit.root, 0, i) for i in pos]
    #     wcnf.append(pysat_atleast_one(roots))

    # root propagation
    # 深さdでrootなら深さd+1でもroot
    # for depth in range(max_depth - 1):
    #     for i in range(n):
    #         root0 = lm.getid(lm.lit.root, depth, i)
    #         root1 = lm.getid(lm.lit.root, depth + 1, i)
    #         wcnf.append(pysat_if(root0, root1))
    #         wcnf.append(pysat_if(-root1, -root0))

    # 深さn-1ではすべての位置がrootとなる
    # for i in range(n):
    #     root = lm.getid(lm.lit.root, n - 1, i)
    #     wcnf.append([root], weight=n + 1)

    # rootでないノードからlinkは出ない
    # for depth in range(n - 1):
    #     for i in range(n):
    #         root0 = lm.getid(lm.lit.root, depth, i)
    #         for j in occ1[text[i]]:
    #             if i == j:
    #                 continue
    #             link_ij = lm.getid(lm.lit.link_to, depth, i, j)
    #             wcnf.append(pysat_if(-root0, -link_ij))

    # i to j at depth>1のリンクが有るならば, k to j at depth-1 のリンクが必ず存在する
    for depth in range(1, max_depth - 1):
        for i in range(n):
            for j in occ1[text[i]]:
                if i == j:
                    continue
                # link_ij = lm.sym(lm.lit.link_to, depth, i, j)
                link_ij = lm.id2sym(lm.getid(lm.lit.link_to, depth, i, j))
                links_to_i = [
                    lm.id2sym(lm.getid(lm.lit.link_to, depth - 1, k, i))
                    for k in occ1[text[i]]
                    if k != i and k != j
                ]
                eq = sympy_if(link_ij, atleast_one(links_to_i))
                wcnf.extend(sympy_cnf_pysat(lm.new_id, eq))

    # rootは唯一つ
    for c in occ1.keys():
        roots = [lm.getid(lm.lit.root, i) for i in occ1[c]]
        wcnf.extend(pysat_equal(lm, 1, roots))

    # 各位置は参照を持つ
    ## rootから
    ## if link i to j at depth, i at depth+1 is root
    # for depth in range(max_depth - 1):
    #     for i in range(n):
    #         root0 = lm.getid(lm.lit.root, depth, i)
    #         for j in occ1[text[i]]:
    #             if i == j:
    #                 continue
    #             root1 = lm.getid(lm.lit.root, depth + 1, j)
    #             ref_ji0 = lm.getid(lm.lit.link_to, depth, i, j)
    #             wcnf.append(pysat_if(ref_ji0, root1))
    #             wcnf.append(pysat_if(ref_ji0, root0))

    # for c, pos in occ.items():
    #     print(c, pos)
    #     for i in pos:
    #         root = lm.id(lm.lit.root, 0, i)
    #         for j in pos:
    #             if i != j:
    #                 link_to = lm.id(lm.lit.link_to, 0, i, j)
    #                 wcnf.append(pysat_if(link_to, root))

    # position jへのリンクの数は高々１
    # iへのリンクの数+depth=0でのrootは必ず１
    for i in range(n):
        rooti = lm.getid(lm.lit.root, i)
        links_i = [
            lm.getid(lm.lit.link_to, depth, j, i)
            for depth in range(max_depth - 1)
            for j in occ1[text[i]]
            if i != j
        ]
        wcnf.extend(pysat_equal(lm, 1, links_i + [rooti]))

    # 各位置はrootもしくは参照される
    # for i in range(n):
    #     root = lm.id(lm.lit.root, 0, i)
    #     links = [lm.id(lm.lit.link_to, 0, j, i) for j in occ[text[i]] if i != j]
    #     wcnf.extend(pysat_equal(lm, 1, links + [root]))

    # solver runs

    # assumptions
    # opt_fbegs = [0, 5, 9, 12, 14, 15, 20]
    # for i in range(n):
    #     fbeg = lm.getid(lm.lit.fbeg, i)
    #     if i in opt_fbegs:
    #         wcnf.append([fbeg])
    #     else:
    #         wcnf.append([-fbeg])

    logger.info(
        f"#literals = {lm.top()}, # hard clauses={len(wcnf.hard)}, # of soft clauses={len(wcnf.soft)}"
    )
    solver = RC2(wcnf)
    sol = solver.compute()
    assert sol is not None
    sold = dict()
    for x in sol:
        sold[abs(x)] = x > 0
    # logger.debug(sol)
    def show_lits(lits):
        for lit in sorted(lits):
            if lit[1]:
                logger.debug(lit[0])

    show_lits(sol2lits2(lm, sold, lm.lit.link_to))
    show_lits(sol2lits2(lm, sold, lm.lit.ref))
    show_lits(sol2lits2(lm, sold, lm.lit.fbeg))
    # show_lits(sol2lits2(lm, sold, lm.lit.root))
    show_sol2(lm, sold, text)
    factors = sol2bidirectional(lm, sold, text)

    logger.debug(factors)
    logger.debug(f"original={text}")
    logger.debug(f"decode={decode(factors).decode('utf8')}")
    assert decode(factors) == text
    return factors


def bidirectional_hdbn(input_file: str) -> BiDirType:
    logger.info("bidirectional_hdbn start")
    cmd = f"cd rustr-master && cargo run --bin optimal_bms -- --input_file {input_file}"
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
    last1 = out.rfind(b"\n")
    last2 = out.rfind(b"\n", 0, last1)
    bd = eval(out[last2:last1])
    return BiDirType(bd)


def bd_info(bd: BiDirType, text: bytes):
    logger.info(f"len={len(bd)}: factors={bd}")
    logger.info(f"len of text = {len(text)}")
    logger.info(f"decode={decode(bd)}")
    logger.info(f"equals original? {decode(bd)==text}")


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

    # factors_hdbn = bidirectional_hdbn(os.path.abspath(args.file))
    # bd_info(factors_hdbn, text)
    # logger.info(f"runtime: {timer()}")

    # text = b"ababab"
    # text = b"abbbaaabb"
    # text = b"abbbaaabb"
    factors_sol = bidirectional_v3(text)
    logger.info(f"runtime: {timer()}")
    # logger.info("solver factors")
    bd_info(factors_sol, text)

    # assert len(factors_sol) == len(factors_hdbn)
