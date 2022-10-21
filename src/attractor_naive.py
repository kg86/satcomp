from typing import List
import time

# Program by Jeffrey Shallit, Dec 12 2020
# https://oeis.org/A339391

from itertools import product, combinations


def blocks_ranges(w):
    blocks = dict()
    for i in range(len(w)):
        for j in range(i + 1, len(w) + 1):
            wij = w[i:j]
            if wij in blocks:
                blocks[wij].append(set(range(i, j)))
            else:
                blocks[wij] = [set(range(i, j))]
    return blocks


def is_attractor(S, w):
    br = blocks_ranges(w)
    for b in br:
        for i in range(len(br[b])):
            if S & br[b][i]:
                break
        else:
            return False
    return True


def lsa(w) -> List[int]:  # length of smallest attractor of w
    for r in range(1, len(w) + 1):
        for s in combinations(range(len(w)), r):
            if is_attractor(set(s), w):
                return list(s)
    return []


# def a(n):  # only search strings starting with 0 by symmetry
#     return max(lsa("0" + "".join(u)) for u in product("01", repeat=n - 1))
# print([a(n) for n in range(1, 20)])

import satcomp.io as io
import satcomp.measure as measure

if __name__ == "__main__":
    parser = io.solver_parser('compute a minimum string attractor naively')
    args = parser.parse_args()
    text = io.read_input(args)

    exp = measure.AttractorExp.create()
    exp.fill_args(args, text)
    exp.algo = 'naive_attractor'

    total_start = time.time()
    l = lsa(text)

    exp.time_total = time.time() - total_start
    exp.output = l
    exp.output_size = len(l)
    io.write_json(args.output, exp)

