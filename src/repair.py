"""Experimental implementation of the RePair grammar-based compressor."""

import argparse
import sys
from typing import List, Tuple


def mostfreq(inttext: List[int]) -> Tuple[Tuple[int, int], int]:
    """Return the most frequent adjacent pair and its frequency."""
    frequencies = dict()
    parity = 0
    for i in range(0, len(inttext) - 1):
        pair = (inttext[i], inttext[i + 1])
        if inttext[i] == inttext[i + 1]:  # do not count character runs doubly like aaa -> 2
            parity += 1
            if parity == 2:
                parity = 0
                continue
        else:
            parity = 0

        if pair in frequencies:
            frequencies[pair] += 1
        else:
            frequencies[pair] = 1
    freqlist = list(frequencies.items())
    freqlist.sort(key=lambda x: x[1])
    return freqlist[-1]


def repair(text: bytes) -> int:
    """Run a simple RePair loop and return the resulting grammar size proxy."""
    print(text)
    inttext = []
    for i in range(0, len(text)):
        el = text[i]
        if isinstance(el, str):
            el = ord(el)
        inttext.append(el)
    FIRST_NONTERMINAL = 256
    last_nonterminal = FIRST_NONTERMINAL
    while True:
        chosen_entry = mostfreq(inttext)
        if chosen_entry[1] == 1:
            break
        chosen_pair = chosen_entry[0]
        for i in range(0, len(inttext) - 1):
            if i >= len(inttext) - 1:
                break
            charA = inttext[i]
            charB = inttext[i + 1]
            pairA = chosen_pair[0]
            pairB = chosen_pair[1]
            if charA == pairA and charB == pairB:
                inttext[i] = last_nonterminal
                inttext.pop(i + 1)
        last_nonterminal += 1
    return 1 + len(inttext) + last_nonterminal - FIRST_NONTERMINAL


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute Minimum SLP.")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
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
    print(repair(text))
