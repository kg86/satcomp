from typing import List, NewType, Tuple

import stralgo

AttractorType = NewType("AttractorType", List[int])


def verify_attractor(text: bytes, attractor: AttractorType) -> bool:
    """
    Verify the attractor.
    """

    # run fast
    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    for b, l in min_substrs:
        lcp_range = stralgo.get_lcprange(lcp, isa[b], l)
        occs = [sa[i] for i in range(lcp_range[0], lcp_range[1] + 1)]
        res = any(occ <= x < (occ + l) for occ in occs for x in attractor)
        if res == False:
            return False
    return True


def main():
    text = """<Y 1874>
<A T. HARDY>
<T Madding Crowd(Penguin 197""".encode(
        "utf8"
    )
    attractor = AttractorType(
        [
            0,
            1,
            4,
            6,
            10,
            11,
            13,
            15,
            17,
            18,
            20,
            21,
            23,
            25,
            26,
            28,
            31,
            33,
            34,
            35,
            36,
            38,
            39,
            40,
            43,
            44,
            45,
            47,
            48,
            49,
        ]
    )
    print(verify_attractor(text, attractor))
    pass


if __name__ == "__main__":
    main()
