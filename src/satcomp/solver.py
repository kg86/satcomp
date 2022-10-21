import enum
from threading import Timer
from pysat.examples.lsu import LSU
from pysat.examples.rc2 import RC2
from pysat.examples.fm import FM
from pysat.formula import WCNF
from typing import Any

class MaxSatType(enum.Enum):
    RC2 = 0
    LSU = 1
    FM = 2
    def __str__(self):
        return str(self.name)


class MaxSatWrapper:
    typ : MaxSatType
    solver : Any
    is_satisfied : bool
    found_optimum : bool
    model : Any
    timeout : int
    logger : Any = None

    def __init__(self, typ : MaxSatType, wcnf : WCNF, timeout : int, verbosity : int, logger = None):
        self.logger = logger
        self.typ = typ
        self.timeout = timeout
        if typ == MaxSatType.RC2:
            self.solver = RC2(wcnf, verbose=verbosity)
        elif typ == MaxSatType.LSU:
            if timeout > 0:
                self.solver = LSU(wcnf, expect_interrupt=True, verbose=verbosity)
            else:
                self.solver = LSU(wcnf, expect_interrupt=False, verbose=verbosity)
        elif typ == MaxSatType.FM:
            self.solver = FM(wcnf, verbose=verbosity)
        else:
            raise Exception(f"unknown MaxSatType: {typ}")

    def compute(self):
        if self.typ == MaxSatType.LSU:
            if self.timeout > 0:
                if self.logger != None:
                    self.logger.info(f"interrupt MAXSAT solver after {self.timeout} seconds")
                timer = Timer(self.timeout, self.interrupt, [self])
                timer.start()
            self.is_satisfied = self.solver.solve()
            self.model = self.solver.model
            self.found_optimum = self.solver.found_optimum()
        elif self.typ == MaxSatType.RC2:
            self.model = self.solver.compute()
            self.is_satisfied = self.model != None
            self.found_optimum = self.is_satisfied
        elif self.typ == MaxSatType.FM:
            self.is_satisfied = self.solver.compute()
            self.model = self.solver.model
            self.found_optimum = self.is_satisfied
    def interrupt(self):
        assert self.typ == MaxSatType.LSU
        if self.logger != None:
            self.logger.info("interrupting MAXSAT-solver...")
        self.solver.interrupt()
