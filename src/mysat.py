"""SAT helper utilities (CNF encodings and a literal manager)."""

from __future__ import annotations

import typing
from collections import defaultdict
from enum import Enum
from typing import Callable, Tuple

from pysat.card import CardEnc, IDPool
from sympy import And, Basic, Not, Or, Symbol
from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue, Equivalent, is_cnf

debug = False


class Literal(Enum):
    """Well-known reserved literals used by `LiteralManager`."""

    true = 1
    false = 2
    auxlit = 3


class LiteralManager:
    """Allocate stable integer IDs for SAT literals and map them to SymPy symbols."""

    def __init__(self) -> None:
        self.vpool = IDPool()
        self.syms: dict[int, Boolean] = dict()
        self.nvar = defaultdict(int)
        self.true = self.newsym(Literal.true)
        self.false = self.newsym(Literal.false)

    def newid(self, *obj: object) -> int:
        """Allocate and return a fresh integer ID for the given key tuple."""
        if len(obj) == 0:
            obj = (Literal.auxlit, self.nvar[Literal.auxlit])
        assert not self.contains(*obj)
        self.nvar[obj[0]] += 1
        return self.vpool.id(obj)

    def getid(self, *obj: object) -> int:
        """Return the existing ID for `obj` (must already be allocated)."""
        assert self.contains(*obj)
        return self.vpool.obj2id[obj]

    def contains(self, *obj: object) -> bool:
        """Return whether an ID has already been allocated for `obj`."""
        return obj in self.vpool.obj2id

    def id2sym(self, id: int) -> Boolean:
        """Return a cached SymPy symbol representing `id`."""
        if id not in self.syms:
            self.syms[id] = Symbol(str(id))
        return self.syms[id]

    def sym2id(self, x: Boolean) -> int:
        """Convert a SymPy literal symbol back to its integer ID."""
        return int(str(x))

    def getsym(self, *opt: object) -> Boolean:
        """Return the SymPy symbol corresponding to a previously allocated key."""
        return self.id2sym(self.getid(*opt))

    def newsym(self, *obj: object) -> Boolean:
        """Allocate a fresh key and return it as a SymPy symbol."""
        return self.id2sym(self.newid(*obj))

    def id2obj(self, id: int) -> object:
        """Return the key tuple associated with `id`."""
        return self.vpool.id2obj[id]

    def id2str(self, id: int) -> str:
        """Return a human-readable string representation of `id`'s key tuple."""
        return str(self.id2obj(id))

    def sym2str(self, x: Boolean) -> str:
        """Return the key tuple string representation for a SymPy symbol."""
        return self.id2str(self.sym2id(x))

    def top(self) -> int:
        """Return the maximum allocated variable ID."""
        return self.vpool.top


def pysat_or(new_var: Callable[[], int], xs: list[int]) -> Tuple[int, list[list[int]]]:
    """Introduce a fresh variable equivalent to OR over `xs` (Tseitin encoding)."""
    nvar = new_var()
    new_clauses = []
    for x in xs:
        new_clauses.append(pysat_if(-nvar, -x))
    new_clause = pysat_if_and_then_or([nvar], xs)
    new_clauses.append(new_clause)
    return nvar, new_clauses


def pysat_and(new_var: Callable[[], int], xs: list[int]) -> Tuple[int, list[list[int]]]:
    """Introduce a fresh variable equivalent to AND over `xs` (Tseitin encoding)."""
    nvar = new_var()
    new_clauses = []
    for x in xs:
        new_clauses.append(pysat_if(nvar, x))
    new_clause = pysat_if_and_then_or([-nvar], [-x for x in xs])
    new_clauses.append(new_clause)
    return nvar, new_clauses


def pysat_atmost(lm: LiteralManager, xs: list[int], bound: int) -> Tuple[int, list[list[int]]]:
    """Encode that the number of true literals in `xs` is at most `bound`."""

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
    """Return a single CNF clause encoding `OR(xs)`."""
    return xs


def pysat_exactlyone(lm: LiteralManager, xs: list[int]) -> Tuple[int, list[list[int]]]:
    """Introduce a fresh variable for the CNF encoding of `exactly_one(xs)`."""
    ex1_clauses = CardEnc.atmost(xs, bound=1, vpool=lm.vpool)
    ex1_clauses.append(pysat_atleast_one(xs))
    res_var, res_clauses = pysat_name_cnf(lm, ex1_clauses)  # type: ignore

    return res_var, res_clauses


def pysat_name_cnf(lm: LiteralManager, xs: list[list[int]]) -> Tuple[int, list[list[int]]]:
    """Introduce a fresh variable equivalent to AND over the CNF `xs`."""
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
    """Return a single clause encoding `(AND(xs)) => (OR(ys))`."""
    return [-x for x in xs] + ys


def pysat_if(x: int, y: int) -> list[int]:
    """Return a single clause encoding `x => y` (i.e., `¬x ∨ y`)."""
    return [-x, y]


def pysat_iff(x: int, y: int) -> list[list[int]]:
    """Return CNF clauses encoding `x <=> y`."""
    return [[-x, y], [x, -y]]


def sympy_atleast_one(lits: list[Boolean]) -> Boolean:
    """Return a SymPy formula encoding `OR(lits)`."""
    return Or(*lits)  # type: ignore


def sympy_atmost_one(lits: list[Boolean]) -> Boolean:
    """Return a SymPy formula encoding `at_most_one(lits)`."""
    n = len(lits)
    return And(*[~lits[i] | ~lits[j] for i in range(n) for j in range(i + 1, n)])  # type: ignore


def sympy_exactly_one(lits: list[Boolean]) -> Boolean:
    """Return a SymPy formula encoding `exactly_one(lits)`."""
    return And(sympy_atleast_one(lits) & sympy_atmost_one(lits))


def sympy_if(x: Boolean, y: Boolean) -> Boolean:
    """Return the implication `x -> y`."""
    return ~x | y


def sympy_iff(x: Boolean, y: Boolean) -> Boolean:
    """Return the bi-implication `x <-> y`."""
    return (~x | y) & (x | ~y)


def sympy_equal(x: Boolean, y: Boolean) -> Boolean:
    """Return the bi-implication `x <-> y`."""
    return (x & y) | (~x & ~y)


def sign_enc(x: int) -> typing.Literal[1, -1]:
    """Convert `x` in {0,1} into {+1,-1}."""
    return 1 if x == 1 else -1


def sign_dec(x: int) -> typing.Literal[0, 1]:
    """Convert a signed pysat literal to a 0/1 sign bit."""
    return 1 if x == 1 else 0


def defcnf(new_var: Callable[[], int], x: Boolean, y: Boolean) -> list[list[int]]:
    """Return a CNF encoding of `x <=> y` (introducing fresh variables as needed)."""
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


def sympy_cnf_pysat(new_var: Callable[[], int], x: Boolean) -> list[list[int]]:
    """
    Convert any sympy equation to cnf of pysat which is boolean equivalent.

    new_var: get a new variable.
    """
    new_formula = []

    def rec(eq: Boolean) -> int:
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
            new_formula.extend(new_clauses)
            return nvar

    x = x.to_nnf()
    if is_cnf(x):
        return cnf_sympy_to_pysat(x)

    z = rec(x)
    new_formula.append([z])
    return new_formula
