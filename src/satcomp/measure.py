from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
import datetime
from pysat.formula import WCNF
from typing import List, NewType
from typing import Union, Any, Tuple
from typing import List, NewType, Dict, Optional, Tuple

import os

@dataclass_json
@dataclass
class MaxSatMeasure:
    is_optimal : bool = False
    is_satisfied: bool = False
    date: str = str(datetime.datetime.now())
    status: str = ""
    algo: str = ""
    file_name: str = "" 
    file_len: int = 0
    time_prep: float = 0
    time_total: float = 0
    timeout : int = 0
    sol_nvars: int = 0
    sol_nhard: int = 0
    sol_nsoft: int = 0
    sol_navgclause: float = 0
    sol_ntotalvars: int = 0
    sol_nmaxclause: int = 0
    output_size: int = 0
    solver: str = ""
    strategy: str = ""

    def fill_args(self, args, text):
        if args.file:
            self.file_name = os.path.basename(args.file)
        self.file_len = len(text)
        if args.solver:
            self.solver = str(args.solver)
        if args.strategy:
            self.strategy = str(args.strategy)
        if args.timeout:
            self.timeout = args.timeout

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
        return MaxSatMeasure()


AttractorType = NewType("AttractorType", List[int])


@dataclass_json
@dataclass
class AttractorExp(MaxSatMeasure):
    output: Union[Any,AttractorType] = field(
            default_factory=lambda: AttractorType([])
            )
    AttractorType([])

    @classmethod
    def create(cls):
        return AttractorExp()


""" BiDirType = [[p0, l0], [p1, l1], ...] represents the string T=T[p0:(p0+l0)]T[p1:(p1+l1)]... """
BiDirType = NewType("BiDirType", List[Tuple[int, int]])

@dataclass_json
@dataclass
class BiDirExp(MaxSatMeasure):
    output_size: int
    output: BiDirType = field(
            default_factory=lambda: BiDirType([])
            )

    @classmethod
    def create(cls):
        return BiDirExp()




"""
type for SLP: represent a partial parse tree via ([i,j,x]) where:
if x == None -> references Node [i,j,None]
if x == Int -> If j-i==1, x is a leaf with symbol x, otherwise is an internal node
whose children are defined via a Dict
SLPNodeType = NewType("SLPNodeType", Tuple[int, int, Optional[int]])
"""
SLPType = NewType(
    "SLPType",
    Tuple[
        Tuple[int, int, Optional[int]],  # root node
        Dict[
            Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]
        ],  # children
    ],
)
#
#
# SLPRuleType = NewType("SLPRuleType", Tuple[int, int, Optional[int]])
#
# SLPType = NewType(
#     "SLPType",
#     Tuple[
#         SLPRuleType,  # root node
#         Dict[
#             SLPRuleType, List[SLPRuleType]
#         ],  # children
#     ],
# )


@dataclass_json
@dataclass
class SLPExp(MaxSatMeasure):
    output : str = ""

    @classmethod
    def create(cls):
        return SLPExp()
