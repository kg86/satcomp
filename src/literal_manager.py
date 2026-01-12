from collections import defaultdict

from pysat.card import IDPool
from sympy import Symbol
from sympy.logic.boolalg import Boolean


class LiteralManager:
    def __init__(self):
        self.vpool = IDPool()
        # self.top_id = 0
        self.nvar = defaultdict(int)
        self.true = self.sym("true")
        self.false = self.sym("false")

    def id(self, *opt: object) -> int:
        return self.vpool.id(opt)

    def sym(self, *opt: object) -> Boolean:
        return Symbol(str(self.id(*opt)))

    def symid(self, x: Boolean) -> int:
        return int(str(x))

    def new_var(self, name: str = "adjlm") -> int:
        self.nvar[name] += 1
        return self.id(name, self.nvar[name])

    def new_var_sym(self, name: str = "adjlm") -> Boolean:
        return Symbol(str(self.new_var(name)))
