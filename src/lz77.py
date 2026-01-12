"""LZ77 factorization (encode/decode) helpers."""

# factors is 2-tuple list
# if factors[i][0] == -1, it represents the character factors[i][1]
# otherwise, it represents the previous appeared substring text[factors[i][0]:factors[i][0]+factors[i][1]]

from typing import List, NewType, Tuple

import stralgo

LZType = NewType("LZType", List[Tuple[int, int]])


def encode(text: bytes) -> LZType:
    """Compute an LZ77 factorization of `text`."""
    res = LZType([])
    n = len(text)
    sa = stralgo.make_sa_MM(text)
    ranka = stralgo.make_isa(sa)
    lcpa = stralgo.make_lcpa_kasai(text, sa, ranka)

    i = 0
    while i < n:
        sai = ranka[i]
        psv, nsv = -1, -1
        psv_len, nsv_len = n, n
        j = sai - 1
        while j >= 0 and psv == -1 and psv_len > 0:
            psv_len = min(psv_len, lcpa[j + 1])
            if sa[j] < i:
                psv = sa[j]
            j -= 1

        j = sai + 1
        while j < n and nsv == -1 and nsv_len > 0:
            nsv_len = min(nsv_len, lcpa[j])
            if sa[j] < i:
                nsv = sa[j]
            j += 1

        psv_len = psv_len if psv != -1 else 0
        nsv_len = nsv_len if nsv != -1 else 0
        if psv_len == 0 and nsv_len == 0:
            res.append((-1, text[i]))
            i += 1
        else:
            # print(i, (psv, psv_len), (nsv, nsv_len))
            prev, prev_len = (psv, psv_len) if psv_len > nsv_len else (nsv, nsv_len)
            assert prev_len > 0
            res.append((prev, prev_len))
            i += prev_len

    return res


def factor_strs(factors: LZType) -> List[bytes]:
    """Return the factor strings corresponding to `factors`."""
    return decode_(factors)[0]


def decode_(factors: LZType) -> Tuple[List[bytes], bytes]:
    """Decode to both factor strings and the reconstructed text."""
    text = []
    res = []
    for factor in factors:
        # print(factor, f"text length = {len(text)}")
        bs = []
        if factor[0] == -1:
            bs.append(factor[1])
            text.append(factor[1])
        else:
            for j in range(factor[1]):
                bs.append(text[factor[0] + j])
                text.append(text[factor[0] + j])
        res.append(bytes(bs))
    return res, bytes(text)


def decode(factors: LZType) -> bytes:
    """Decode and return the reconstructed text."""
    return decode_(factors)[1]


def equal(text: bytes, f1: LZType, f2: LZType) -> bool:
    """Return whether two factorizations represent the same substrings in `text`."""
    # verify factor form
    if len(f1) != len(f2):
        return False
    for i in range(len(f1)):
        if f1[i][0] == -1 and f2[i][0] == -1 and f1[i][1] == f2[i][1]:
            pass
        elif f1[i][0] >= 0 and f2[i][0] >= 0 and f1[i][1] == f2[i][1]:
            if text[f1[i][0] : f1[i][0] + f1[i][1]] != text[f2[i][0] : f2[i][0] + f2[i][1]]:
                print(f"i={i}, f1={f1[i]}, f2={f2[i]}")
                return False
        else:
            print(f"i={i}, f1={f1[i]}, f2={f2[i]}")
            return False

    return True
