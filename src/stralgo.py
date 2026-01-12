from typing import Iterable, List, Optional, Tuple

from tqdm import tqdm


def make_sa_MM(text: str) -> list[int]:
    """
    Make sufix array by Manber and Myers algorithm.
    """
    if not (isinstance(text, str) or isinstance(text, bytes)):
        assert "text must be str or bytes"
    n = len(text)
    d = 1
    rank = [ord(text[i]) if isinstance(text, str) else text[i] for i in range(n)]
    rank2 = [0 for _ in range(n)]
    sa = list(range(n))
    while d < n:

        def at(i: int) -> tuple[int, int]:
            key1 = rank[i]
            key2 = rank[i + d] if (i + d) < len(rank) else -1
            return (key1, key2)

        sa.sort(key=at)
        # print_sa(text, sa)

        rank2[0] = 0
        for i in range(1, n):
            rank2[sa[i]] = rank2[sa[i - 1]] + (0 if at(sa[i - 1]) == at(sa[i]) else 1)
        for i in range(n):
            rank[i] = rank2[i]
        d *= 2

    return sa


def get_lcp(text: str, i: int, j: int) -> int:
    n = len(text)
    l = 0
    while i < n and j < n:
        if text[i] != text[j]:
            break
        i += 1
        j += 1
        l += 1
    return l


def make_isa(sa: list[int]) -> list[int]:
    """
    Make inverse suffix array.
    """
    n = len(sa)
    isa = [0 for _ in range(n)]
    for i in range(n):
        isa[sa[i]] = i
    return isa


def make_lcpa_kasai(text: str, sa: list[int], isa: list[int] | None = None) -> list[int]:
    """
    Make longest common prefix array by Kasai algorithm.
    """
    n = len(text)
    if isa is None:
        isa = make_isa(sa)

    lcp = [0 for _ in range(n)]
    l = 0
    for i in range(n):
        if isa[i] != 0:
            lcp[isa[i]] = l + get_lcp(text, sa[isa[i]] + l, sa[isa[i] - 1] + l)
            l = lcp[isa[i]]
        l = max(0, l - 1)
    return lcp


def get_bwt(text: str, sa: list[int]) -> list[str]:
    n = len(text)
    res = []

    for i in range(n):
        res.append(text[sa[i] - 1])
    return res


def get_lcprange(lcp: List[int], i: int, least_lcp: Optional[int] = None) -> Tuple[int, int]:
    """
    Compute the maximum range lcp[j1:j2] such that
    least_lcp <= lcp[j] for j in [j1+1:j2]
    let least_lcp be lcp[i] if it is None.
    """

    n = len(lcp)
    if least_lcp is None:
        least_lcp = lcp[i]
    b, e = i, i
    while 0 < b and least_lcp <= lcp[b]:
        b -= 1
    while e < n - 1 and least_lcp <= lcp[e + 1]:
        e += 1
    return (b, e)


def maximal_repeat(text: str, sa: list[int], lcp: list[int]) -> list[tuple[int, int]]:
    n = len(text)
    res = []

    def is_bwt_distinct(lcp_range: tuple[int, int]) -> bool:
        if sa[lcp_range[0]] == 0:
            return True
        for i in range(lcp_range[0] + 1, lcp_range[1] + 1):
            if sa[i] == 0:
                return True
            if text[sa[i - 1] - 1] != text[sa[i] - 1]:
                return True
        return False

    lcp_range_prev = (0, 0)
    for i in range(1, n):
        lcp_range_cur = get_lcprange(lcp, i, lcp[i])
        # print(i, lcp_range_cur, is_bwt_distinct(lcp_range_cur))
        if lcp[i] > 0 and lcp_range_prev != lcp_range_cur and is_bwt_distinct(lcp_range_cur):
            res.append((sa[i], lcp[i]))
        lcp_range_prev = lcp_range_cur
    return res


def occ_pos_naive(text: str, pattern: str) -> list[int]:
    res = []
    beg = 0
    occ = text.find(pattern, beg)
    while occ >= 0:
        res.append(occ)
        beg = occ + 1
        occ = text.find(pattern, beg)
    return res


def num_occ(text: str, pattern: str) -> int:
    """
    Compute the number of occurrences of the pattern in text.
    """
    return len(occ_pos_naive(text, pattern))


def substr(text: str) -> List[str]:
    n = len(text)
    res = []
    for i in range(n):
        for j in range(i, n):
            res.append(text[i : j + 1])
    return res


def minimum_substr_naive(text: str) -> List[Tuple[int, int]]:
    """
    Compute the set of (b, l) s.t. text[b:b+l] is a minimum substring.
    A minimum substring x is a substring that the #occ of x is
    different from #occ of x[1:] and also #occ of x[:-1].
    """
    n = len(text)
    set_substr = set(text[i : j + 1] for i in range(n) for j in range(i, n))
    res = []
    for substr in set_substr:
        nocc = num_occ(text, substr)
        nocc_ptruncate = num_occ(text, substr[1:])
        nocc_struncate = num_occ(text, substr[:-1])
        if nocc != nocc_ptruncate and nocc != nocc_struncate:
            res.append((text.find(substr), len(substr)))
    return res


def minimum_right_substr(text: str) -> list[tuple[int, int]]:
    sa = make_sa_MM(text)
    isa = make_isa(sa)
    lcp = make_lcpa_kasai(text, sa, isa)
    return minimum_right_substr_sa(text, sa, isa, lcp)


def minimum_right_substr_sa(text: str, sa: List[int], isa: List[int], lcp: List[int]) -> List[Tuple[int, int]]:
    """
    Compute the set of (b, l) s.t. text[b:b+l] is a minimum right substring
    A minimum right substring x is a substring that the #occ of x is
    different from #occ of #occ of x[:-1].
    """
    n = len(text)
    res: List[Tuple[int, int]] = [(sa[0], 1)]
    already_computed = set()
    for i in tqdm(range(1, n)):
        if lcp[i] == 0:
            if sa[i] + 1 <= n:
                res.append((sa[i], 1))
            continue
        if lcp[i - 1] == lcp[i]:
            continue
        lcp_range = get_lcprange(lcp, i)
        # print(i, lcp_range)
        if (lcp_range, lcp[i]) in already_computed:
            continue
        cur = lcp_range[0]
        while cur <= lcp_range[1]:
            # text[sa[cur]:sa[cur]+lcp[i]+1] is a minimum right substring
            lcp_range_sub = get_lcprange(lcp, cur, lcp[i] + 1)
            assert lcp_range[0] <= lcp_range_sub[0] <= lcp_range_sub[1] <= lcp_range[1]
            if sa[cur] + lcp[i] + 1 > n:
                cur += 1
                continue

            already_computed.add((lcp_range, lcp[i]))
            res.append((sa[cur], lcp[i] + 1))
            assert cur < lcp_range_sub[1] + 1
            cur = lcp_range_sub[1] + 1

    return res


def minimum_substr(text: str) -> list[tuple[int, int]]:
    sa = make_sa_MM(text)
    isa = make_isa(sa)
    lcp = make_lcpa_kasai(text, sa, isa)
    return minimum_substr_linear(text, sa, isa, lcp)


def minimum_substr_sa(text: str, sa: List[int], isa: List[int], lcp: List[int]) -> List[Tuple[int, int]]:
    """
    Compute the set of (b, l) s.t. text[b:b+l] is a minimum substring
    A minimum substring x is a substring that the #occ of x is
    different from #occ of x[1:] and also #occ of x[:-1].
    """
    return minimum_substr_linear(text, sa, isa, lcp)


def minimum_substr_square(text: str, sa: List[int], isa: List[int], lcp: List[int]) -> List[Tuple[int, int]]:
    """
    Compute the set of (b, l) s.t. text[b:b+l] is a minimum substring
    A minimum substring x is a substring that the #occ of x is
    different from #occ of x[1:] and also #occ of x[:-1].
    """
    n = len(text)
    res: List[Tuple[int, int]] = [(sa[0], 1)]
    already_computed = set()
    for i in tqdm(range(1, n)):
        if lcp[i] == 0:
            res.append((sa[i], 1))
            continue
        if lcp[i - 1] == lcp[i]:
            continue
        lcp_range = get_lcprange(lcp, i)
        assert (lcp_range[1] - lcp_range[0] + 1) > 1
        # for k in range(lcp_range[0], lcp_range[1]):
        #     assert text[sa[k] : sa[k] + lcp[i]] == text[sa[k + 1] : sa[k + 1] + lcp[i]]
        if (lcp_range, lcp[i]) in already_computed:
            continue
        cur = lcp_range[0]
        while cur <= lcp_range[1]:
            # let substr = text[sa[i]:sa[i]+lcp[i]]
            # substr is an explicit node of the suffix tree
            # so, #substr and #substr+text[sa[cur]+lcp[cur] must be different
            lcp_range_sub = get_lcprange(lcp, cur, lcp[i] + 1)
            len_lrange_sub = lcp_range_sub[1] - lcp_range_sub[0] + 1
            assert lcp_range[0] <= lcp_range_sub[0] <= lcp_range_sub[1] <= lcp_range[1]
            if sa[cur] + lcp[i] + 1 > n:
                cur += 1
                continue

            # let substr = text[sa[cur]:sa[cur]+lcp[i]]
            # check whether #substr equals #substr[1:] or not.
            lcp_range_sub2 = get_lcprange(lcp, isa[sa[cur] + 1], lcp[i])
            len_lrange_sub2 = lcp_range_sub2[1] - lcp_range_sub2[0] + 1
            if len_lrange_sub != len_lrange_sub2:
                already_computed.add((lcp_range, lcp[i]))
                res.append((sa[cur], lcp[i] + 1))
            assert cur < lcp_range_sub[1] + 1
            cur = lcp_range_sub[1] + 1
    # print("#res=", len(res))
    return res


def minimum_substr_linear(text: str, sa: List[int], isa: List[int], lcp: List[int]) -> List[Tuple[int, int]]:
    """
    Compute the set of (b, l) s.t. text[b:b+l] is a minimum substring
    A minimum substring x is a substring that the #occ of x is
    different from #occ of x[1:] and also #occ of x[:-1].
    """
    n = len(text)

    class Node:
        """
        Node of suffix trees.
        `depth`: the depth of node.
        [`begin`, `end`]: the interval of the suffix array prefixed by the string of the node.
        """

        def __init__(self, depth: int, begin: int, end: int):
            self.depth = depth
            self.begin = begin
            self.end = end

        def show(self) -> str:
            return f"(depth={self.depth}, begin={self.begin}, end={self.end})"

    root = Node(0, 0, -1)
    leaf = Node(n - sa[0], 0, -1)
    path = [root, leaf]  # post order traversal
    parent_c = []  # [(u, c, v)]: (u, v) is a pair of parent and child, and c is the first label on the edge
    for i in range(1, n):
        child = None
        while lcp[i] < path[-1].depth:
            node = path.pop()
            node.end = i - 1
            if child:
                c = text[sa[child.begin] + node.depth]
                parent_c.append((node, c, child))
            child = node
        if lcp[i] > path[-1].depth:
            assert child is not None
            # create internal node
            node = Node(lcp[i], child.begin, -1)
            path.append(node)
        if child:
            c = text[sa[child.begin] + path[-1].depth]
            parent_c.append((path[-1], c, child))
        leaf = Node(n - sa[i], i, -1)
        path.append(leaf)
    child = None
    while len(path) > 0:
        node = path.pop()
        node.end = n - 1
        if child:
            c = text[sa[child.begin] + node.depth]
            parent_c.append((node, c, child))
        child = node

    # compute minimum substring
    res = []
    for parent, c, child in parent_c:
        if (
            parent == root
            or child.end - child.begin != isa[sa[child.end] + 1] - isa[sa[child.begin] + 1]
            or ((isa[sa[child.end] + 1] + 1 < n) and lcp[isa[sa[child.end] + 1] + 1] >= parent.depth)
            or lcp[isa[sa[child.begin] + 1]] >= parent.depth
        ):
            res.append((sa[child.begin], parent.depth + 1))

    return res


def substr_cover(text: str, sa: list[int], lcp: list[int], isa: list[int], b: int, l: int) -> set[int]:
    """
    return occ of text[b:b+l] in text.
    """
    lcp_range = get_lcprange(lcp, isa[b], l)
    print(lcp_range)
    cover = set()
    for sai in range(lcp_range[0], lcp_range[1] + 1):
        for i in range(sa[sai], sa[sai] + l):
            cover.add(i)
    return cover


def print_sa(text: str, sa: list[int]) -> None:
    n = len(text)
    for i in range(n):
        print("\t".join(map(str, [i, sa[i], text[sa[i] :]])))


def print_sa_lcp(text: str, sa: list[int], lcp: list[int]) -> None:
    n = len(text)
    for i in range(n):
        print(
            "\t".join(
                map(
                    str,
                    [
                        i,
                        sa[i],
                        lcp[i],
                        text[sa[i] - 1],
                        text[sa[i] : sa[i] + lcp[i] + 3],
                    ],
                )
            )
        )


def verify_sa(text: str, sa: list[int]):
    n = len(text)
    for i in range(1, n):
        # print('{} < {} ?'.format(text[sa[i - 1]:], text[sa[i]:]))
        assert text[sa[i - 1] :] < text[sa[i] :]


def gen_binary(n: int) -> Iterable[str]:
    """
    Generates all binary strings of length `n`.
    """
    if n == 1:
        yield "a"
        yield "b"
    elif n > 1:
        for suf in gen_binary(n - 1):
            yield suf + "a"
            yield suf + "b"
    else:
        assert False


if __name__ == "__main__":
    text = "bananabanana$"
    text = "banana"
    sa = make_sa_MM(text)
    verify_sa(text, sa)

    isa = make_isa(sa)
    lcp = make_lcpa_kasai(text, sa, isa)
    print_sa_lcp(text, sa, lcp)

    print(substr_cover(text, sa, lcp, isa, 3, 1))
    mstr_naive = sorted([text[b : b + l] for b, l in minimum_substr_naive(text)])
    mstr_square = sorted([text[b : b + l] for b, l in minimum_substr_square(text, sa, isa, lcp)])
    mstr_linear = sorted([text[b : b + l] for b, l in minimum_substr_linear(text, sa, isa, lcp)])
    print("minimum substr naive")
    print(mstr_naive)
    print(mstr_square)
    print(mstr_linear)
    assert len(mstr_naive) == len(mstr_square)
    assert len(mstr_naive) == len(mstr_linear)
    for x, y, z in zip(mstr_naive, mstr_square, mstr_linear):
        assert x == y
        assert x == z
