from typing import List
from typing_extensions import Self
from sympy.logic.boolalg import (
    BooleanFalse,
    BooleanTrue,
    to_cnf,
    Implies,
    Equivalent,
    BooleanFunction,
    Boolean,
    is_cnf,
    is_nnf,
    ITE,
)
from sympy import Basic, Symbol, Or, And, Not, Xor

from mysat import (
    LiteralManager,
    literal_sympy_to_pysat,
    sympy_atleast_one,
    sympy_atmost_one,
    sympy_equal,
    sympy_exactly_one,
)


def half_adder(a: Boolean, b: Boolean) -> list[Boolean, Boolean]:  # type: ignore
    # [carry, sum]
    return a & b, a ^ b  # type: ignore


def full_adder(a: Boolean, b: Boolean, c: Boolean) -> list[Boolean, Boolean]:  # type: ignore
    c1, s1 = half_adder(a, b)
    c2, s2 = half_adder(s1, c)
    return c1 | c2, s2  # type: ignore


def msb(x: int) -> int:
    digit = 0
    while x > 0:
        digit += 1
        x //= 2
    return digit


class LogEncoding:
    def __init__(
        self, lm: LiteralManager, consts: List[Boolean], assumps: List[int], digit: int
    ):
        self.lm = lm
        self.consts = consts
        self.assumps = assumps
        self.digit = digit
        self.ds = [Symbol(str(self.lm.newid("LE"))) for _ in range(self.digit)]

    def __str__(self):
        return str(list(reversed(self.ds)))

    def set(self, i: int, eq: Boolean):
        """
        Add constraint that self.ds[i] = eq.
        """
        self.consts.append(Equivalent(self.ds[i], eq))  # type: ignore

    def set_num(self, num: int):
        """
        Add constraint that self represents the number `num`.
        """
        assert num < 2**self.digit
        for i in range(self.digit):
            if num % 2 == 0:
                self.consts.append(~self.ds[i])
            else:
                self.consts.append(self.ds[i])
            num //= 2

    def new(self) -> Self:
        return LogEncoding(self.lm, self.consts, self.assumps, self.digit)

    def get_num(self, assign):
        return sum(d << i for i, d in enumerate(self.sol(assign)))

    def sol(self, assign):
        ds = []
        for d in self.ds:
            lit = literal_sympy_to_pysat(d)
            ds.append(assign[lit])
        return ds

    def show(self, assign):
        return str(list(reversed(self.sol(assign))))

    @classmethod
    def const(
        cls, lm: LiteralManager, consts: List[Boolean], assumps: List[int], x: int
    ) -> Self:
        res = LogEncoding(lm, consts, assumps, msb(x))
        for i in range(res.digit):
            res.consts.append(res.ds[i] if (x >> i) % 2 == 1 else ~res.ds[i])
        return res

    def constant(self, x: int) -> Self:
        digit = msb(x)
        res = LogEncoding(self.lm, self.consts, self.assumps, digit)

        for i in range(digit):
            res.consts.append(res.ds[i] if (x >> i) % 2 == 1 else ~res.ds[i])
        return res

    def add_assumps(self, i: int, is_true: Boolean):
        sign = 1 if is_true else -1
        self.assumps.append(sign * literal_sympy_to_pysat(self.ds[i]))

    def set_all(self, target):
        x, y = (self, target) if self.digit <= target.digit else (target, self)
        for i in range(x.digit):
            x.set(i, y.ds[i])
        for i in range(x.digit, y.digit):
            y.set(i, self.lm.false)

    def equal_i(self, i: int, eq: Boolean):
        return Equivalent(self.ds[i], eq)

    def equal_all(self, target):
        x, y = (self, target) if self.digit <= target.digit else (target, self)
        res = And()
        for i in range(x.digit):
            res &= x.equal_i(i, y.ds[i])
        for i in range(x.digit, y.digit):
            res &= y.equal_i(i, self.lm.false)
        return res

    def lshift(self, k: int) -> Self:
        res = LogEncoding(self.lm, self.consts, self.assumps, self.digit + k)
        for i in range(k):
            res.set(i, self.lm.false)
            # res.ds[i] = False
        for i in range(k, res.digit):
            res.set(i, self.ds[i - k])
        return res

    def add(self, target: Self) -> Self:
        x, y = (self, target) if self.digit <= target.digit else (target, self)
        if x.digit == 0:
            return y
        res = LogEncoding(self.lm, self.consts, self.assumps, y.digit + 1)
        c, s = half_adder(self.ds[0], target.ds[0])
        res.set(0, s)
        for i in range(1, x.digit):
            c, s = full_adder(self.ds[i], target.ds[i], c)
            res.set(i, s)
        for i in range(x.digit, y.digit):
            c, s = half_adder(y.ds[i], c)
            res.set(i, s)
        res.set(res.digit - 1, c)
        return res

    def mul(self, target: Self) -> Self:
        x, y = (self, target) if self.digit <= target.digit else (target, self)
        if x.digit == 0:
            return x
        # ss = [LogEncoding(y.lm, y.consts, y.assumps, y.digit)]
        ss = [y.new()]
        self.consts.append(
            ITE(x.ds[0], ss[0].equal_all(y), ss[0].equal_all(x.constant(0)))
        )
        for i in range(1, x.digit):
            ss.append(LogEncoding(x.lm, x.consts, y.assumps, y.digit + i + 1))
            self.consts.append(
                ITE(
                    x.ds[i],
                    ss[i].equal_all(y.lshift(i).add(ss[i - 1])),
                    ss[i].equal_all(ss[i - 1]),
                )
            )
        return ss[-1]

    def leq(self, target: Self) -> Boolean:
        def leq_(x: Boolean, y: Boolean) -> Boolean:
            # return (~x & ~y) | y
            return ~x | y

        def lt(x: Boolean, y: Boolean) -> Boolean:
            return ~x & y

        if self.digit == 0:
            return self.lm.true
        if target.digit == 0:
            # always false since self.digit >0 and target.digit = 0
            zero = And()
            for i in range(self.digit):
                zero &= sympy_equal(self.ds[i], self.lm.false)
            nv = self.lm.newsym("leq")
            self.consts.append(sympy_equal(nv, zero))
            return nv
            # return self.lm.false

        tpre = self.lm.newsym("leq")
        self.consts.append(sympy_equal(tpre, leq_(self.ds[0], target.ds[0])))
        for i in range(1, min(self.digit, target.digit)):
            eq = lt(self.ds[i], target.ds[i]) | (
                sympy_equal(self.ds[i], target.ds[i]) & tpre
            )
            tpre = self.lm.newsym("leq")
            # print("const", sympy_equal(tpre, eq))
            self.consts.append(sympy_equal(tpre, eq))

        # digit
        res = self.lm.newsym("leq")
        if self.digit < target.digit:
            eq = tpre | sympy_atleast_one(target.ds[self.digit :])
            self.consts.append(sympy_equal(res, eq))
            # self.consts.append(eq | tpre)
        elif self.digit >= target.digit:
            eq = tpre & And(*[~self.ds[i] for i in range(target.digit, self.digit)])
            self.consts.append(sympy_equal(res, eq))
            # for i in range(target.digit, self.digit):
            #     self.consts.append(~self.ds[i])
            # self.consts.append(tpre)
        return res

        # assert isinstance(eq, Or)
        # nvar = new_var()
        # literals = [rec(clause) for clause in eq.args]
        # new_clauses = []
        # for literal in literals:
        #     new_clauses.append([nvar, -literal])
        # new_clause = [-nvar] + literals
        # new_clauses.append(new_clause)
        # if debug:
        #     print(f"{nvar}={Or(*literals)}, cnf={And(*new_clauses)}")
        # # assert is_nnf(And(*new_clauses))
        # new_formula.extend(new_clauses)
        # return nvar


def sumlits(
    lm: LiteralManager,
    consts: list[Boolean | Any],
    assumps: list[int],
    lits: list[Boolean],
) -> LogEncoding:
    res = LogEncoding(lm, consts, assumps, msb(len(lits)))
    if len(lits) == 0:
        pass
    elif len(lits) == 1:
        res.set(0, lits[0])
    elif len(lits) == 2:
        c1, s1 = half_adder(lits[0], lits[1])
        res.set(0, s1)
        res.set(1, c1)
    elif len(lits) == 3:
        c1, s1 = full_adder(lits[0], lits[1], lits[2])
        res.set(0, s1)
        res.set(1, c1)
    else:
        x = sumlits(lm, consts, assumps, lits[: len(lits) // 2])
        y = sumlits(lm, consts, assumps, lits[len(lits) // 2 :])
        res.set_all(x.add(y))
    return res


if __name__ == "__main__":
    w, x, y, z = Symbol("w"), Symbol("x"), Symbol("y"), Symbol("z")
    print(sympy_atleast_one([x & y, y, z]))
    print(sympy_atmost_one([x, y, z]))
    print(sympy_exactly_one([x, y, z]))

    # adder
    print("adder")
    c, s = half_adder(y, z)
    print((c, s))
    print(full_adder(w, x, c))

    # # Direct Encoding
    # print("Direct Encoding")
    # digit = 2
    # xs = [Symbol("x" + str(d)) for d in range(digit)]
    # ys = [Symbol("y" + str(d)) for d in range(digit)]
    # xd = DirectEncoding(xs)
    # yd = DirectEncoding(ys)
    # print(xd)
    # print(xd.add(yd))
    # print(xd.lshift(1))
    # print(xd.rshift(1))

    print("Log Encoding")
    digit = 2
    # xd = to_de(0)
    # xs = [Symbol("x" + str(d)) for d in range(digit)]
    consts = []
    assumps = []
    lm = LiteralManager()
    xd = LogEncoding(lm, consts, assumps, digit)
    # ys = [Symbol("y" + str(d)) for d in range(digit)]
    yd = LogEncoding(lm, consts, assumps, digit)
    print("xd", xd)
    print("yd", yd)
    print("xd + yd", xd.add(yd))
    print("xd * yd", xd.mul(yd))

    zd = LogEncoding(lm, consts, assumps, digit)
    # print(zd.equal(xd.add(yd)) & ~xd.equal(yd))
