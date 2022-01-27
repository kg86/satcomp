from pysat.card import IDPool


class LiteralManager:
    def __init__(self):
        self.vpool = IDPool()
        # self.top_id = 0
        self.nvar = defaultdict(int)
        self.true = self.sym("true")
        self.false = self.sym("false")

    def id(self, *opt) -> int:
        return self.vpool.id(opt)

    def sym(self, *opt) -> Boolean:
        return Symbol(str(self.id(*opt)))

    def symid(self, x: Boolean) -> int:
        return int(str(x))

    def new_var(self, name: str = "adjlm") -> int:
        self.nvar[name] += 1
        return self.id(name, self.nvar[name])
        # return self.vpool.id((name, self.nvar[name]))

    def new_var_sym(self, name: str = "adjlm") -> Boolean:
        return Symbol(str(self.new_var(name)))
