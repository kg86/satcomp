# Compute permutation by using SAT solver

from collections import namedtuple
import sys
import argparse
from logging import getLogger, DEBUG, INFO, StreamHandler

from left_refference import LRLiteralManaer

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


from dataclasses import dataclass

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2

from mysat import *


@dataclass
class Literal:
    link: str = "link"
    can_link: str = "can_link"
    start: str = "start"


class PermLiteralManager(LiteralManager):
    def __init__(self):
        self.lit = Literal()
        self.validf = dict()
        super().__init__()

    def id(self, *opt) -> int:
        res = super().newid(*opt)
        if opt[0] in self.validf:
            self.validf[opt[0]](opt)
        return res


def pairs(n: int):
    for i in range(n):
        for j in range(n):
            if i != j:
                yield (i, j)


def lit_by_id(sol: list[int], id: int):
    return sol[id - 1] > 0


def show_sol(lm: PermLiteralManager, sol: list[int], n: int):
    for i in range(n):
        lid = lm.id(lm.lit.start, i)
        if lit_by_id(sol, lid):
            logger.debug(lm.id2str(lid))

    for step in range(n - 1):
        link = None
        for (i, j) in pairs(n):
            lid = lm.id(lm.lit.link, step, i, j)
            lit = lit_by_id(sol, lid)
            if lit:
                logger.debug(lm.id2str(lid))


def permutation():
    n = 4
    wcnf = WCNF()
    lm = PermLiteralManager()

    def valid_link(opt):
        assert len(opt) == 4
        assert 0 <= opt[1] < n - 1
        assert 0 <= opt[2], opt[3] < n
        assert opt[2] != opt[3]

    def valid_can_link(opt):
        assert len(opt) == 3
        assert 0 <= opt[1] < n - 1
        assert 0 <= opt[2] < n

    lm.validf[lm.lit.link] = valid_link
    lm.validf[lm.lit.can_link] = valid_can_link
    wcnf.append([lm.sym2id(lm.true)])
    wcnf.append([lm.sym2id(lm.false)])

    starts = [lm.id(lm.lit.start, i) for i in range(n)]
    wcnf.extend(
        CardEnc.equals(starts, bound=1, encoding=EncType.pairwise, top_id=lm.top() + 1)
    )
    for i in range(n):
        if_body = lm.getsym(lm.lit.start, i)
        if_then = sympy_exactly_one(
            [lm.getsym(lm.lit.link, 0, i, j) for j in range(n) if i != j]
        )
        wcnf.extend(sympy_cnf_pysat(lm.newid, sympy_if(if_body, if_then)))
        wcnf.append(pysat_if(lm.id(lm.lit.start, i), -lm.id(lm.lit.can_link, 0, i)))

    for step in range(n - 2):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                #
                link0 = lm.id(lm.lit.link, step, i, j)
                can_link0 = lm.id(lm.lit.can_link, step, j)
                can_link1 = lm.id(lm.lit.can_link, step + 1, j)
                wcnf.append(pysat_if(link0, -can_link1))
                link0_sym = lm.id2sym(link0)
                # if link i to j at step, then link j to k at step+1.
                for (k1, k2) in pairs(n):
                    if k1 != j:
                        link1 = lm.id(lm.lit.link, step + 1, k1, k2)
                        wcnf.append(pysat_if(link0, -link1))
            can_link0 = lm.id(lm.lit.can_link, step, i)
            can_link1 = lm.id(lm.lit.can_link, step + 1, i)
            wcnf.append(pysat_if(-can_link0, -can_link1))
    for step in range(n - 1):
        all_links = [lm.id(lm.lit.link, step, i, j) for (i, j) in pairs(n)]
        wcnf.extend(
            CardEnc.equals(
                all_links, bound=1, encoding=EncType.pairwise, top_id=lm.top() + 1
            )
        )
        for (i, j) in pairs(n):
            link0 = lm.id(lm.lit.link, step, i, j)
            can_link0 = lm.id(lm.lit.can_link, step, j)
            wcnf.append(pysat_if(-can_link0, -link0))

    solver = RC2(wcnf)
    sol = solver.compute()
    assert sol is not None
    print(sol)
    show_sol(lm, sol, n)


if __name__ == "__main__":
    permutation()
