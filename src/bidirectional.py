from dataclasses import dataclass
from typing import List, NewType, Tuple

# BiDirType = [[p0, l0], [p1, l1], ...] represents the string T=T[p0:(p0+l0)]T[p1:(p1+l1)]...
BiDirType = NewType("BiDirType", List[Tuple[int, int]])


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
    sol_nvars: int
    sol_nhard: int
    sol_nsoft: int
    factor_size: int
    factors: BiDirType

    @classmethod
    def create(cls):
        return BiDirExp(
            str(datetime.datetime.now()), "", "", "", 0, 0, 0, 0, 0, 0, 0, BiDirType([])
        )


def bd_info(bd: BiDirType, text: bytes) -> str:

    return "\n".join(
        [
            f"len={len(bd)}: factors={bd}",
            f"len of text = {len(text)}",
            f"decode={decode(bd)}",
            f"equals original? {decode(bd)==text}",
        ]
    )


def decode_len(factors: BiDirType) -> int:
    """
    Computes the length of decoded string from a given bidirectional scheme.
    """
    res = 0
    for f in factors:
        res += 1 if f[0] == -1 else f[1]
    return res


def decode(factors: BiDirType) -> bytes:
    """
    Computes the decoded string from a given bidirectional scheme.
    """
    n = decode_len(factors)
    res = [-1 for _ in range(n)]
    nfs = len(factors)
    fbegs = []
    is_decoded = [False for _ in factors]
    n_decoded_fs = 0
    pos = 0
    for i, f in enumerate(factors):
        fbegs.append(pos)
        if f[0] == -1:
            is_decoded[i] = True
            n_decoded_fs += 1
            res[pos] = f[1]
            pos += 1
        else:
            pos += f[1]

    while n_decoded_fs < nfs:
        for fi, f in enumerate(factors):
            refi, reflen = f
            if is_decoded[fi]:
                continue
            pos = fbegs[fi]
            count = 0
            for j in range(reflen):
                if res[refi + j] != -1:
                    count += 1
                    res[pos + j] = res[refi + j]
            if reflen == count:
                is_decoded[fi] = True
                n_decoded_fs += 1

    return bytes(res)


if __name__ == "__main__":
    factors_naive = BiDirType([(13, 8), (0, 11), (-1, 98), (-1, 97)])
    factors_sol = BiDirType([(8, 8), (13, 8), (-1, 97), (-1, 98), (16, 3)])
    print(decode(factors_naive))
    print(decode(factors_sol))
