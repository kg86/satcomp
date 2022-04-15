from dataclasses import dataclass
from dataclasses_json import dataclass_json
import datetime
from attractor import AttractorType
from pysat.formula import WCNF


@dataclass_json
@dataclass
class AttractorExp:
    date: str
    status: str
    algo: str
    file_name: str
    file_len: int
    time_prep: float
    time_total: float
    sol_nvars: int
    sol_nhard: int
    sol_nsoft: int
    sol_navgclause: float
    sol_ntotalvars: int
    sol_nmaxclause: int
    factor_size: int
    factors: AttractorType

    def fill(self, wcnf: WCNF):
        self.sol_nvars = wcnf.nv
        self.sol_nhard = len(wcnf.hard)
        self.sol_nsoft = len(wcnf.soft)
        max_clause = max(wcnf.hard, key=lambda item: len(item))
        self.sol_nmaxclause = len(max_clause)

        var_in_clause_sum=0
        for i in range(0,len(wcnf.hard)):
            var_in_clause_sum+=len(wcnf.hard[i])
        avg_var_in_clause = var_in_clause_sum / len(wcnf.hard)
        self.sol_ntotalvars = var_in_clause_sum
        self.sol_navgclause = avg_var_in_clause

    @classmethod
    def create(cls):
        return AttractorExp(
     date =                  str(datetime.datetime.now()),
     status =                "",
     algo =                  "",
     file_name =             "",
     file_len =              0,
     time_prep =             0,
     time_total =            0,
     sol_nvars =             0,
     sol_nhard =             0,
     sol_nsoft =             0,
     factor_size =           0,
     sol_navgclause = 0.0,
     sol_ntotalvars = 0,
     sol_nmaxclause = 0,
     factors =               AttractorType([]),
        )






