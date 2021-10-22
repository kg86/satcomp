from dataclasses import dataclass
from typing import NewType

BiDirType = NewType("BiDirType", list[tuple[int, int]])


from dataclasses_json import dataclass_json
import datetime


@dataclass_json
@dataclass
class BiDirExp:
    date: str
    status: str
    algo: str
    file_name: str
    file_len: int
    time_prep: float
    time_total: float
    bd_size: int
    bd_factors: BiDirType
    sol_nvars: int
    sol_nhard: int
    sol_nsoft: int

    # def __post_init__(self):
    #     if self.time_prep is None:
    #         self.time_prep = 0
    #     if self.time_total is None:
    #         self.time_tota = 0
    #     if self.file_name is None:
    #         self.file_name = ""
    #     if self.file_len is None:
    #         self.file_len = 0
    #     if self.bd_size is None:
    #         self.bd_size = 0
    #     if self.bd_factors is None:
    #         self.bd_factors = BiDirType([])
    #     if self.sol_nvars is None:
    #         self.sol_nvars = 0
    #     if self.sol_nhard is None:
    #         self.sol_nhard = 0
    #     if self.sol_nsoft is None:
    #         self.sol_nsoft = 0

    @classmethod
    def create(cls):
        return BiDirExp(
            str(datetime.datetime.now()), "", "", "", 0, 0, 0, 0, BiDirType([]), 0, 0, 0
        )


# @dataclass_json
# @dataclass
# class BiDirExp:
#     time_prep: float = 0
#     time_total: float = 0
#     file_name: str = ""
#     file_len: int = 0
#     bd_size: int = 0
#     bd_factors: BiDirType = BiDirType([])
#     sol_nvars: int = 0
#     sol_nhard: int = 0
#     sol_nsoft: int = 0

# def __init__(self):
#     self.prep = 0.0
#     self.sol_nvars = 0
#     self.sol_nhard = 0
#     self.sol_nsoft = 0


def decode_len(factors: BiDirType) -> int:
    res = 0
    for f in factors:
        res += 1 if f[0] == -1 else f[1]
    return res


def decode(factors: BiDirType) -> bytes:
    n = decode_len(factors)
    # none = 'x'
    res = [-1 for _ in range(n)]
    i = 0
    fs = [list(f) for f in factors]
    fbegs = []
    for f in fs:
        fbegs.append(i)
        if f[0] == -1:
            res[i] = f[1]
            i += 1
        else:
            i += f[1]

    # logger.debug(f"fbegs={fbegs}")
    for step in range(n):
        found = sum(1 for i in res if i != -1)
        # logger.debug(f"step={step}: found={found}/{n}")
        if found == n:
            break
        for fi, f in enumerate(fs):
            refi, reflen = f
            if refi == -1:
                continue
            i = fbegs[fi]
            count = 0
            for j in range(reflen):
                if res[refi + j] != -1:
                    count += 1
                    res[i + j] = res[refi + j]
            if reflen == count:
                fs[fi][0] = -1

    # logger.debug(f"decode={res}")
    return bytes(res)


if __name__ == "__main__":
    factors_naive = BiDirType([[13, 8], [0, 11], [-1, 98], [-1, 97]])
    factors_sol = [[8, 8], [13, 8], [-1, 97], [-1, 98], [16, 3]]
    print(decode(factors_naive))
    print(decode(factors_sol))
