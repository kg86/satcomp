from typing import AnyStr, Tuple


def make_sa_MM(text):
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
        at = lambda i: (rank[i], rank[i + d] if (i + d) < len(rank) else -1)
        sa.sort(key=at)
        # print_sa(text, sa)

        rank2[0] = 0
        for i in range(1, n):
            rank2[sa[i]] = rank2[sa[i - 1]] + (0 if at(sa[i - 1]) == at(sa[i]) else 1)
        for i in range(n):
            rank[i] = rank2[i]
        d *= 2

    return sa


def get_lcp(text, i, j):
    n = len(text)
    l = 0
    while i < n and j < n:
        if text[i] != text[j]:
            break
        i += 1
        j += 1
        l += 1
    return l


def make_isa(sa):
    """
    Make inverse suffix array
    """
    n = len(sa)
    isa = [0 for _ in range(n)]
    for i in range(n):
        isa[sa[i]] = i
    return isa


def make_lcpa_kasai(text, sa, isa=None):
    """
    Make longest common prefix array by Kasai algorithm.
    """
    n = len(text)
    if isa == None:
        isa = make_isa(sa)

    lcp = [0 for _ in range(n)]
    l = 0
    for i in range(n):
        if isa[i] != 0:
            lcp[isa[i]] = l + get_lcp(text, sa[isa[i]] + l, sa[isa[i] - 1] + l)
            l = lcp[isa[i]]
        l = max(0, l - 1)
    return lcp


def get_bwt(text, sa):
    n = len(text)
    res = []

    for i in range(n):
        res.append(text[sa[i] - 1])
    return res
    # if isinstance(text, str):
    #   return ''.join(res)
    # return res


def get_lcprange(lcp, i, least_lcp=None):
    """
    Compute the maximum range lcp[j1:j2] such that
    lcp[j] = lcp[i] for j in [j1:j2]
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


def maximal_repeat(text, sa, lcp):
    n = len(text)
    res = []

    def is_bwt_distinct(lcp_range):
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
        if (
            lcp[i] > 0
            and lcp_range_prev != lcp_range_cur
            and is_bwt_distinct(lcp_range_cur)
        ):
            res.append((sa[i], lcp[i]))
        lcp_range_prev = lcp_range_cur
    return res


def occ_pos_naive(text, pattern):
    res = []
    beg = 0
    occ = text.find(pattern, beg)
    while occ >= 0:
        res.append(occ)
        beg = occ + 1
        occ = text.find(pattern, beg)
    return res


def num_occ(text, pattern):
    """
    return number of occ of pattern in text.
    """
    return len(occ_pos_naive(text, pattern))


def substr(text: AnyStr) -> list[AnyStr]:
    n = len(text)
    res = []
    for i in range(n):
        for j in range(i, n):
            res.append(text[i : j + 1])
    return res


def minimum_substr_naive(text) -> list[Tuple[int, int]]:
    """
    return set of (b, l) s.t. text[b:b+l] is a minimum substring.
    a minimum substring x is a substring that the #occ of x is
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


def minimum_substr(text):
    sa = make_sa_MM(text)
    isa = make_isa(sa)
    lcp = make_lcpa_kasai(text, sa, isa)
    return minimum_substr_sa(text, sa, isa, lcp)


def minimum_substr_sa(text, sa, isa, lcp):
    """
    return set of (b, l) s.t. text[b:b+l] is a minimum substring
    a minimum substring x is a substring that the #occ of x is
    different from #occ of x[1:] and also #occ of x[:-1].
    """
    n = len(text)
    res = [(sa[0], 1)]
    already_computed = set()
    for i in range(1, n):
        if lcp[i] == 0:
            res.append((sa[i], 1))
            continue
        if lcp[i - 1] == lcp[i]:
            continue
        lcp_range = get_lcprange(lcp, i)
        assert (lcp_range[1] - lcp_range[0] + 1) > 1
        for k in range(lcp_range[0], lcp_range[1]):
            assert text[sa[k] : sa[k] + lcp[i]] == text[sa[k + 1] : sa[k + 1] + lcp[i]]
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


def substr_cover(text, sa, lcp, isa, b, l):
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


def print_sa(text, sa):
    n = len(text)
    for i in range(n):
        print("\t".join(map(str, [i, sa[i], text[sa[i] :]])))


def print_sa_lcp(text, sa, lcp):
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


def verify_sa(text, sa):
    n = len(text)
    for i in range(1, n):
        # print('{} < {} ?'.format(text[sa[i - 1]:], text[sa[i]:]))
        assert text[sa[i - 1] :] < text[sa[i] :]


def test():
    text = open("cantrbry/grammar.lsp", "r").read()
    print(len(text))
    sa = make_sa_MM(text)
    isa = make_isa(sa)
    lcp = make_lcpa_kasai(text, sa, isa)
    verify_sa(text, sa)
    print("valid")

    with open("hoge.sa", "w") as f:
        for i in range(len(text)):
            f.write("{} {}\n".format(i, text[sa[i] : sa[i] + lcp[i] + 3]))


if __name__ == "__main__":
    # test()
    # import sys
    # sys.exit(0)
    text = "bananabanana$"
    # text = open('fuga.txt', 'r').read()
    sa = make_sa_MM(text)
    verify_sa(text, sa)

    isa = make_isa(sa)
    lcp = make_lcpa_kasai(text, sa, isa)
    print_sa_lcp(text, sa, lcp)

    # print([(text[b:b+l], (b,l)) for b, l in mrepeat])
    print(substr_cover(text, sa, lcp, isa, 3, 1))
    print("minimum substr naive")
    print([text[b : b + l] for b, l in minimum_substr_naive(text)])
    # print([text[b:b+l] for b, l in minimum_substr_sa(text, sa, isa, lcp)])
    print([text[b : b + l] for b, l in minimum_substr_sa(text, sa, isa, lcp)])
