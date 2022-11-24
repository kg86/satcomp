import enum
from threading import Timer
from pysat.examples.lsu import LSU
from pysat.examples.rc2 import RC2
from pysat.examples.fm import FM
from pysat.formula import WCNF
from typing import Any

class MaxSatStrategy(enum.Enum):
    RC2 = 0
    LSU = 1
    FM = 2
    def __str__(self):
        return str(self.name)

class SolverType(enum.Enum):
    Glucose4 = 0
    Cadical = 1
    def __str__(self):
        return str(self.name)
    def toString(self):
        if self == SolverType.Glucose4:
            return "g4"
        else:
            return "cd"

class MaxSatWrapper:
    strategy : MaxSatStrategy
    solver : Any
    is_satisfied : bool
    found_optimum : bool
    model : Any
    timeout : int
    logger : Any = None

    def __init__(self, strategy : MaxSatStrategy, solvertype : SolverType, wcnf : WCNF, timeout : int, verbosity : int, logger = None):
        self.logger = logger
        self.strategy = strategy
        self.timeout = timeout
        hasTimeout = timeout > 0

        if strategy == MaxSatStrategy.RC2:
            incremental = False
            if solvertype == SolverType.Glucose4:
                incremental = True
            self.solver = RC2(wcnf, solver=solvertype.toString(), verbose=verbosity, incr=incremental)
        elif strategy == MaxSatStrategy.LSU:
            self.solver = LSU(wcnf, solver=solvertype.toString(), expect_interrupt=hasTimeout, verbose=verbosity)
        elif strategy == MaxSatStrategy.FM:
            self.solver = FM(wcnf, verbose=verbosity, solver=solvertype.toString())
        else:
            raise Exception(f"unknown MaxSatStrategy: {strategy}")

    def compute(self):
        if self.strategy == MaxSatStrategy.LSU:
            if self.timeout > 0:
                if self.logger != None:
                    self.logger.info(f"interrupt MAXSAT solver after {self.timeout} seconds")
                timer = Timer(self.timeout, self.interrupt, [])
                timer.start()
            self.is_satisfied = self.solver.solve()
            self.model = self.solver.model
            self.found_optimum = self.solver.found_optimum()
        elif self.strategy == MaxSatStrategy.RC2:
            self.model = self.solver.compute()
            self.is_satisfied = self.model != None
            self.found_optimum = self.is_satisfied
        elif self.strategy == MaxSatStrategy.FM:
            self.is_satisfied = self.solver.compute()
            self.model = self.solver.model
            self.found_optimum = self.is_satisfied

    def interrupt(self):
        assert self.strategy == MaxSatStrategy.LSU
        if self.logger != None:
            self.logger.info("interrupting MAXSAT-solver...")
        self.solver.interrupt()
