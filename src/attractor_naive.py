import argparse
import sys
import time
from itertools import combinations

# Program by Jeffrey Shallit, Dec 12 2020
# https://oeis.org/A339391


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


def lsa(w):  # length of smallest attractor of w
    for r in range(1, len(w) + 1):
        for s in combinations(range(len(w)), r):
            if is_attractor(set(s), w):
                return r


# def a(n):  # only search strings starting with 0 by symmetry
#     return max(lsa("0" + "".join(u)) for u in product("01", repeat=n - 1))
# print([a(n) for n in range(1, 20)])


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum String Attractors.")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--output", type=str, help="output file", default="")
    # parser.add_argument(
    #     "--contains",
    #     nargs="+",
    #     type=int,
    #     help="list of text positions that must be included in the string attractor, starting with index 1",
    #     default=[],
    # )
    # parser.add_argument(
    #     "--size",
    #     type=int,
    #     help="exact size or upper bound of attractor size to search",
    #     default=0,
    # )
    # parser.add_argument(
    #     "--algo",
    #     type=str,
    #     help=(
    #         "[min: find a minimum string attractor, exact/atmost: find a string "
    #         "attractor whose size is exact/atmost SIZE]"
    #     ),
    # )
    # parser.add_argument(
    #     "--log_level",
    #     type=str,
    #     help="log level, DEBUG/INFO/CRITICAL",
    #     default="CRITICAL",
    # )
    args = parser.parse_args()
    if (
        args.file == ""
        and args.str == ""
        # or args.algo not in ["exact", "atmost", "min"]
        # or (args.algo in ["exact", "atmost"] and args.size <= 0)
        # or (args.log_level not in ["DEBUG", "INFO", "CRITICAL"])
    ):
        parser.print_help()
        sys.exit()

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.str != "":
        text = args.str
    else:
        text = open(args.file, "rb").read()
    total_start = time.time()
    l = lsa(text)
    print(f"smallest string attractor: {l}")
    print(f"time to compute: {time.time() - total_start}")
