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
    depth_ref: str = "depth_ref"


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


def sol2refs(lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes):
    n = len(text)
    occ = make_occa(text)
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


def show_sol2(lm: BiDirLiteralManager, sol: dict[int, bool], text: bytes):
    n = len(text)
    occ = make_occa(text)
    pinfo = defaultdict(list)
    refs = sol2refs(lm, sol, text)

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
    refs = sol2refs(lm, sol, text)
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


def make_occa2(text: bytes) -> dict[int, list[int]]:
    match2 = defaultdict(list)
    for i in range(len(text) - 1):
        for j in range(i):
            if text[i : i + 2] == text[j : j + 2]:
                match2[i].append(j)
                match2[j].append(i)
    return match2


def bidirectional(text: bytes):
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
    for depth in range(max_depth - 1):
        for i in range(n):
            lits.append(lm.id(lm.lit.depth_ref, depth, i))

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
                # logger.debug(
                #     f"if link {i} to {j} at depth {depth}, then ref {j} to {i}"
                # )
                link_ij = lm.getid(lm.lit.link_to, depth, i, j)
                wcnf.append(pysat_if(link_ij, ref_ji0))
            # if i > 0 and j > 0 and text[i - 1] != text[j - 1]:
            if i == 0 or j == 0 or text[i - 1] != text[j - 1]:
                # logger.debug(f"if ref {j} to {i}, then fbeg {j}")
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

    # i to j at depth>1のリンクが有るならば, k to j at depth-1 のリンクが必ず存在する
    logger.debug("link is connectted, and it forms tree structure")
    for depth in range(1, max_depth - 1):
        if depth % 30 == 0:
            logger.debug(f"depth {depth}/{max_depth}")
        for i in range(n):
            for j in occ1[text[i]]:
                if i == j:
                    continue
                # link_ij = lm.id2sym(lm.getid(lm.lit.link_to, depth, i, j))
                # links_to_i = [
                #     lm.id2sym(lm.getid(lm.lit.link_to, depth - 1, k, i))
                #     for k in occ1[text[i]]
                #     if k != i and k != j
                # ]
                # eq = sympy_if(link_ij, atleast_one(links_to_i))
                # wcnf.extend(sympy_cnf_pysat(lm.new_id, eq))

                # if link_ij, then sum(links_to_i) >= 1
                link_ij = lm.getid(lm.lit.link_to, depth, i, j)
                dref_i = lm.getid(lm.lit.depth_ref, depth - 1, i)
                wcnf.append(pysat_if(link_ij, dref_i))
                # links_to_i = [
                #     lm.getid(lm.lit.link_to, depth - 1, k, i)
                #     for k in occ1[text[i]]
                #     if k != i and k != j
                # ]
                # wcnf.append(pysat_if_and_then_or([link_ij], links_to_i))
                # wcnf.append([-link_ij] + links_to_i)

    # rootは唯一つ
    for c in occ1.keys():
        roots = [lm.getid(lm.lit.root, i) for i in occ1[c]]
        wcnf.extend(pysat_equal(lm, 1, roots))

    # 参照はたかだか１つ
    # rootなら参照はしない
    logger.debug("# of referrences is only one")
    for i in range(n):
        refs = [lm.getid(lm.lit.ref, i, j) for j in occ1[text[i]] if i != j]
        root_i = lm.getid(lm.lit.root, i)
        wcnf.extend(CardEnc.atmost(refs, bound=1, vpool=lm.vpool))
        # wcnf.extend(pysat_equal(lm, 1, refs + [root_i]))
        for j in occ1[text[i]]:
            if i == j:
                continue
            ref_ij = lm.getid(lm.lit.ref, i, j)
            link_ij = lm.getid(lm.lit.link_to, 0, i, j)
            wcnf.append(pysat_if(root_i, -ref_ij))
            wcnf.append(pysat_if(-root_i, -link_ij))

    # ここの処理が重たい
    # position jへのリンクの数は高々１
    # iへのリンクの数+depth=0でのrootは必ず１
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
    # depth refの定義
    for depth in range(max_depth - 1):
        if depth % 30 == 0:
            logger.debug(f"depth {depth}/{max_depth}")
        for i in range(n):
            for j in occ1[text[i]]:
                if i == j:
                    continue
                link_ij = lm.getid(lm.lit.link_to, depth, i, j)
                dref_j = lm.getid(lm.lit.depth_ref, depth, j)
                # jへのリンクがあればdrefをtrueにする
                wcnf.append(pysat_if(link_ij, dref_j))
            links_to_i = [
                lm.getid(lm.lit.link_to, depth, j, i) for j in occ1[text[i]] if i != j
            ]
            # 各深さでiへのリンクは高々１つ
            wcnf.extend(CardEnc.atmost(links_to_i, bound=1, vpool=lm.vpool))

            dref_i = lm.getid(lm.lit.depth_ref, depth, i)
            dref_i_sym = lm.id2sym(dref_i)
            # dref_iなら、１つのリンクを持つ
            wcnf.append(pysat_if_and_then_or([dref_i], links_to_i))
            # links_to_i_sym = [lm.id2sym(x) for x in links_to_i]
            # if_then = atleast_one(links_to_i_sym)
            # wcnf.extend(sympy_cnf_pysat(lm.new_id, sympy_if(dref_i_sym, if_then)))

            # if dref_iがfalseなら, iへのすべてのリンクはfalse
            if_then, clauses = pysat_and(lm.new_id, [-x for x in links_to_i])
            # logger.debug(clauses)
            wcnf.extend(clauses)
            wcnf.append(pysat_if(-dref_i, if_then))
            # if_then = And(*[~lm.id2sym(x) for x in links_to_i])
            # wcnf.extend(sympy_cnf_pysat(lm.new_id, sympy_if(~dref_i_sym, if_then)))
    for i in range(n):
        dref_i = [
            lm.getid(lm.lit.depth_ref, depth, i) for depth in range(max_depth - 1)
        ]
        root_i = lm.getid(lm.lit.root, i)
        wcnf.extend(pysat_equal(lm, 1, dref_i + [root_i]))

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
    factors_sol = bidirectional(text)
    logger.info(f"runtime: {timer()}")
    # logger.info("solver factors")
    bd_info(factors_sol, text)

    # assert len(factors_sol) == len(factors_hdbn)
