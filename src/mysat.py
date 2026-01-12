from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Callable, Tuple

from pysat.card import CardEnc, IDPool
from sympy import And, Basic, Not, Or, Symbol
from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue, Equivalent, is_cnf

debug = False

# ClauseType = NewType("ClauseType", List[int])
# CNFType = NewType("ClauseType", List[Clause])


class Literal(Enum):
    true = 1
    false = 2
    auxlit = 3


class LiteralManager:
    def __init__(self, lits=Literal):
        self.lits = lits
        # self.lits = Literal
        self.vpool = IDPool()
        self.syms: dict[int, Boolean] = dict()
        self.nvar = defaultdict(int)
        self.true = self.newsym(self.lits.true)
        self.false = self.newsym(self.lits.false)

    def newid(self, *obj: object) -> int:
        if len(obj) == 0:
            # obj = ("auxlit", self.nvar["auxlit"])
            obj = (self.lits.auxlit, self.nvar[self.lits.auxlit])
        assert obj[0] in self.lits
        assert not self.contains(*obj)
        self.nvar[obj[0]] += 1
        return self.vpool.id(obj)

    def getid(self, *obj: object) -> int:
        assert self.contains(*obj)
        return self.vpool.obj2id[obj]

    def contains(self, *obj: object) -> bool:
        return obj in self.vpool.obj2id

    def id2sym(self, id: int) -> Boolean:
        if id not in self.syms:
            self.syms[id] = Symbol(str(id))
        return self.syms[id]

    def sym2id(self, x: Boolean) -> int:
        return int(str(x))

    def getsym(self, *opt: object) -> Boolean:
        return self.id2sym(self.getid(*opt))

    def newsym(self, *obj: object) -> Boolean:
        return self.id2sym(self.newid(*obj))

    def id2obj(self, id: int) -> object:
        return self.vpool.id2obj[id]

    def id2str(self, id: int) -> str:
        return str(self.id2obj(id))

    def sym2str(self, x: Boolean) -> str:
        return self.id2str(self.sym2id(x))

    def top(self) -> int:
        return self.vpool.top


# def pysat_or(new_var: Callable[[], int], xs: list[int]) -> Tuple[int, list[list[int]]]:
#     nvar = new_var()
#     new_clauses = []
#     # nvar <=> or(xs)
#     # (nvar => or(xs)) and (or(xs) => nvar)
#     # ((not nvar) OR or(xs)) and ((not or(xs)) OR nvar)
#     # ((not nvar) OR or(xs)) and ((and(not x)) OR nvar)
#     # ((not nvar) OR or(xs)) and (and (not x OR nvar))
#     new_clauses.append([-nvar] + xs)
#     for x in xs:
#         new_clauses.append([nvar,-x])
#     nvar, new_clauses


def pysat_or(new_var: Callable[[], int], xs: list[int]) -> Tuple[int, list[list[int]]]:
    nvar = new_var()
    new_clauses = []
    for x in xs:
        new_clauses.append(pysat_if(-nvar, -x))
    new_clause = pysat_if_and_then_or([nvar], xs)
    new_clauses.append(new_clause)
    return nvar, new_clauses


def pysat_and(new_var: Callable[[], int], xs: list[int]) -> Tuple[int, list[list[int]]]:
    nvar = new_var()
    new_clauses = []
    for x in xs:
        new_clauses.append(pysat_if(nvar, x))
    new_clause = pysat_if_and_then_or([-nvar], [-x for x in xs])
    new_clauses.append(new_clause)
    return nvar, new_clauses


def pysat_atmost(lm: LiteralManager, xs: list[int], bound: int) -> Tuple[int, list[list[int]]]:
    """
    Create a literal and clauses such that the number of true literals in `xs` is at most `bound`.
    """

    atmost_clauses = CardEnc.atmost(xs, bound=bound, vpool=lm.vpool)

    xs = []
    new_clauses = []
    for clause in atmost_clauses:
        nvar, clauses = pysat_or(lm.newid, clause)
        new_clauses.extend(clauses)
        xs.append(nvar)
    nvar, clauses = pysat_and(lm.newid, xs)
    new_clauses.extend(clauses)
    return nvar, new_clauses


def pysat_atleast_one(xs: list[int]) -> list[int]:
    return xs


# def pysat_exactlyone(lm: LiteralManager, xs: list[int]) -> Tuple[int, list[list[int]]]:
#     new_clauses = pysat_atleast_one(xs)
#     nvar, clauses = pysat_atmost(lm, xs, bound=1)
#     new_clauses.extend(clauses)
#     return pysat_and(lm.newid, new_clauses)


def pysat_exactlyone(lm: LiteralManager, xs: list[int]) -> Tuple[int, list[list[int]]]:
    ex1_clauses = CardEnc.atmost(xs, bound=1, vpool=lm.vpool)
    # _, ex1_clauses = pysat_atmost(lm, xs, bound=1)
    # res_clauses = []
    ex1_clauses.append(pysat_atleast_one(xs))
    res_var, res_clauses = pysat_name_cnf(lm, ex1_clauses)  # type: ignore
    # res_clauses.extend(ex1_clauses)

    return res_var, res_clauses
    # return pysat_name_cnf(lm, ex1_clauses)


def pysat_name_cnf(lm: LiteralManager, xs: list[list[int]]) -> Tuple[int, list[list[int]]]:
    res_clauses = []
    ex1_vars = []
    for clause in xs:
        nvar, or_clauses = pysat_or(lm.newid, clause)
        ex1_vars.append(nvar)
        res_clauses.extend(or_clauses)
    res_var, clauses = pysat_and(lm.newid, ex1_vars)
    res_clauses.extend(clauses)
    return res_var, res_clauses


def pysat_if_and_then_or(xs: list[int], ys: list[int]) -> list[int]:
    return [-x for x in xs] + ys


def pysat_if(x: int, y: int) -> list[int]:
    return [-x, y]


def pysat_iff(x: int, y: int) -> list[list[int]]:
    return [[-x, y], [x, -y]]


def sympy_atleast_one(lits: list[Boolean]) -> Boolean:
    return Or(*lits)  # type: ignore


def sympy_atmost_one(lits: list[Boolean]) -> Boolean:
    n = len(lits)
    return And(*[~lits[i] | ~lits[j] for i in range(n) for j in range(i + 1, n)])  # type: ignore


def sympy_exactly_one(lits: list[Boolean]) -> Boolean:
    return And(sympy_atleast_one(lits) & sympy_atmost_one(lits))


def sympy_if(x: Boolean, y: Boolean) -> Boolean:
    """
    x -> y
    """
    return ~x | y


def sympy_iff(x: Boolean, y: Boolean) -> Boolean:
    """
    x <-> y
    """
    return (~x | y) & (x | ~y)


def sympy_equal(x: Boolean, y: Boolean) -> Boolean:
    """
    x <-> y
    """
    return (x & y) | (~x & ~y)


def sign_enc(x):
    """
    Convert x= 1 or 0 to 1 or -1.
    """
    return 1 if x == 1 else -1


def sign_dec(x):
    return 1 if x == 1 else 0


def defcnf(new_var: Callable[[], int], x: Boolean, y: Boolean) -> list[list[int]]:
    return sympy_cnf_pysat(new_var, Equivalent(x, y))  # type: ignore


def literal_sympy_to_pysat(x: Boolean | Basic) -> int:
    """
    Convert sympy literal to pysat literal.
    sympy literal must be represented by integer
    """
    assert isinstance(x, Symbol) or isinstance(x, Not)
    if isinstance(x, Not):
        return -int(str(x.args[0]))
    else:
        return int(str(x))


def cnf_sympy_to_pysat(x: Boolean | Basic) -> list[list[int]]:
    """
    Convert cnf of sympy to cnf of pysat.
    x is sympy cnf formula whose literal is number
    """

    def convert_clause(y: Boolean | Basic) -> list[int]:
        if isinstance(y, Symbol) or isinstance(y, Not):
            return [literal_sympy_to_pysat(y)]
        elif isinstance(y, Or):
            return [literal_sympy_to_pysat(z) for z in y.args]
        raise Exception("Only Or, Symbol, Not are allowed")

    if isinstance(x, BooleanFalse):
        return [[1], [-1]]
    elif isinstance(x, BooleanTrue):
        return []
    elif not isinstance(x, And):
        return [convert_clause(x)]

    return [convert_clause(clause) for clause in x.args]


def sympy_cnf_pysat(new_var: Callable[[], int], x: Boolean | Basic) -> list[list[int]]:
    """
    Convert any sympy equation to cnf of pysat which is boolean equivalent.

    new_var: get a new variable.
    """
    new_formula = []

    def rec(eq: Boolean | Basic) -> int:
        if isinstance(eq, Symbol) or isinstance(eq, Not):
            return literal_sympy_to_pysat(eq)
        elif isinstance(eq, And):
            nvar = new_var()
            literals = [rec(clause) for clause in eq.args]
            new_clauses = []
            for literal in literals:
                new_clauses.append([-nvar, literal])
            new_clause = [nvar] + [-literal for literal in literals]
            new_clauses.append(new_clause)
            if debug:
                print(f"{nvar}={And(*literals)}, cnf={And(*new_clauses)}")  # type: ignore
            new_formula.extend(new_clauses)

            return nvar
        else:
            assert isinstance(eq, Or)
            nvar = new_var()
            literals = [rec(clause) for clause in eq.args]
            new_clauses = []
            for literal in literals:
                new_clauses.append([nvar, -literal])  # type: ignore
            new_clause = [-nvar] + literals
            new_clauses.append(new_clause)
            if debug:
                print(f"{nvar}={Or(*literals)}, cnf={And(*new_clauses)}")  # type: ignore
            # assert is_nnf(And(*new_clauses))
            new_formula.extend(new_clauses)
            return nvar

    x = x.to_nnf()
    if is_cnf(x):
        return cnf_sympy_to_pysat(x)

    z = rec(x)
    new_formula.append([z])
    return new_formula
