import datetime
from dataclasses import dataclass
from typing import Dict, List, NewType, Optional, Tuple

from dataclasses_json import dataclass_json
from pysat.formula import WCNF

# type for SLP: represent a partial parse tree via ([i,j,x]) where:
# if x == None -> references Node [i,j,None]
# if x == Int -> If j-i==1, x is a leaf with symbol x, otherwise is an internal node
# whose children are defined via a Dict
# SLPNodeType = NewType("SLPNodeType", Tuple[int, int, Optional[int]])

# SLPType = NewType(
#     "SLPType",
#     Tuple[SLPNodeType, Dict[SLPNodeType, Optional[Tuple[SLPNodeType, SLPNodeType]]]],
# )

SLPType = NewType(
    "SLPType",
    Tuple[
        Tuple[int, int, Optional[int]],  # root node
        Dict[
            Tuple[int, int, Optional[int]], List[Tuple[int, int, Optional[int]]]
        ],  # children
    ],
)


@dataclass_json
@dataclass
class SLPExp:
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
    factors: str

    def fill(self, wcnf: WCNF):
        self.sol_nvars = wcnf.nv
        self.sol_nhard = len(wcnf.hard)
        self.sol_nsoft = len(wcnf.soft)
        max_clause = max(wcnf.hard, key=lambda item: len(item))
        self.sol_nmaxclause = len(max_clause)

        var_in_clause_sum = 0
        for i in range(0, len(wcnf.hard)):
            var_in_clause_sum += len(wcnf.hard[i])
        avg_var_in_clause = var_in_clause_sum / len(wcnf.hard)
        self.sol_ntotalvars = var_in_clause_sum
        self.sol_navgclause = avg_var_in_clause

    @classmethod
    def create(cls):
        return SLPExp(
            date=str(datetime.datetime.now()),
            status="",
            algo="",
            file_name="",
            file_len=0,
            time_prep=0,
            time_total=0,
            sol_nvars=0,
            sol_nhard=0,
            sol_nsoft=0,
            factor_size=0,
            sol_navgclause=0.0,
            sol_ntotalvars=0,
            sol_nmaxclause=0,
            factors="",
        )


# def slp_info(slp: SLPType, text: bytes) -> str:
#     return "\n".join(
#         [
#             f"len={len(slp)}: factors={slp}",
#             f"len of text = {len(text)}",
#             f"decode={decode(slp)}",
#             f"equals original? {decode(slp)==text}",
#         ]
#     )


# def decode(slp: SLPType) -> bytes:
#     """
#     Computes the decoded string from a given SLP
#     """
#     (root, slpmap) = slp

#     def decode_aux(root: SLPNodeType) -> List[int]:
#         res = []
#         (i, j, ref) = root
#         if j - i == 1:
#             res.append(ref)
#         else:
#             children = slpmap[root]
#             assert children != None
#             if ref is None:
#                 res += decode_aux(children[0])
#                 res += decode_aux(children[1])
#             else:
#                 n = SLPNodeType((ref, ref + j - i, None))
#                 res += decode_aux(n)
#         return []

#     res = []
#     if root != None:
#         res = decode_aux(root)
#     return bytes(res)


if __name__ == "__main__":
    pass
    # factors_naive = SLPType([(13, 8), (0, 11), (-1, 98), (-1, 97)])
    # factors_sol = SLPType([(8, 8), (13, 8), (-1, 97), (-1, 98), (16, 3)])
    # print(decode(factors_naive))
    # print(decode(factors_sol))
