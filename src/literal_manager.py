"""A small literal manager wrapper around `pysat`'s `IDPool`."""

from collections import defaultdict

from pysat.card import IDPool
from sympy import Symbol
from sympy.logic.boolalg import Boolean


class LiteralManager:
    """Allocate SAT variable IDs and provide SymPy symbol views."""

    def __init__(self):
        self.vpool = IDPool()
        # self.top_id = 0
        self.nvar = defaultdict(int)
        self.true = self.sym("true")
        self.false = self.sym("false")

    def id(self, *opt: object) -> int:
        """Return a stable integer ID for the given key tuple."""
        return self.vpool.id(opt)

    def sym(self, *opt: object) -> Boolean:
        """Return a SymPy symbol representing the ID of `opt`."""
        return Symbol(str(self.id(*opt)))

    def symid(self, x: Boolean) -> int:
        """Convert a SymPy symbol back to its integer ID."""
        return int(str(x))

    def new_var(self, name: str = "adjlm") -> int:
        """Allocate a fresh variable ID under the given name prefix."""
        self.nvar[name] += 1
        return self.id(name, self.nvar[name])

    def new_var_sym(self, name: str = "adjlm") -> Boolean:
        """Allocate a fresh variable and return it as a SymPy symbol."""
        return Symbol(str(self.new_var(name)))
