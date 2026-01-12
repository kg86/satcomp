"""Naive (exponential-time) algorithms for minimum string attractors."""

import argparse
import sys
import time
from itertools import combinations

# Program by Jeffrey Shallit, Dec 12 2020
# https://oeis.org/A339391


def blocks_ranges(w: bytes) -> dict[bytes, list[set[int]]]:
    """Return, for each substring, the list of occurrence position-sets."""
    blocks = dict()
    for i in range(len(w)):
        for j in range(i + 1, len(w) + 1):
            wij = w[i:j]
            if wij in blocks:
                blocks[wij].append(set(range(i, j)))
            else:
                blocks[wij] = [set(range(i, j))]
    return blocks


def is_attractor(S: set[int], w: bytes) -> bool:
    """Check whether `S` hits every distinct substring occurrence of `w`."""
    br = blocks_ranges(w)
    for b in br:
        for i in range(len(br[b])):
            if S & br[b][i]:
                break
        else:
            return False
    return True


def lsa(w: bytes) -> int:  # length of smallest attractor of w
    """Return the length of a smallest string attractor (brute force)."""
    for r in range(1, len(w) + 1):
        for s in combinations(range(len(w)), r):
            if is_attractor(set(s), w):
                return r
    raise RuntimeError("Unreachable: no attractor size found")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute Minimum String Attractors.")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--output", type=str, help="output file", default="")
    args = parser.parse_args()
    if args.file == "" and args.str == "":
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
