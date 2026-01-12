import argparse
import copy
import json
import os
import sys
import time
from logging import CRITICAL, DEBUG, INFO, Formatter, StreamHandler, getLogger

from attractor_bench_format import AttractorExp

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
FORMAT = "[%(lineno)s - %(funcName)10s() ] %(message)s"
formatter = Formatter(FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

##############################################################################################
# Code by rici
# https://stackoverflow.com/questions/14900693/enumerate-all-full-labeled-binary-tree
#
# A very simple representation for Nodes. Leaves are anything which is not a Node.


class Node(object):
    def __init__(self, left: str, right: str) -> None:
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return "(%s %s)" % (self.left, self.right)


def enum_ordered(labels: str) -> str | Node:
    if len(labels) == 1:
        yield labels[0]
    else:
        for i in range(1, len(labels)):
            for left in enum_ordered(labels[:i]):
                for right in enum_ordered(labels[i:]):
                    yield Node(left, right)


##############################################################################################


def minimize_tree(root: str | Node, nodedic: dict[str | tuple[int, int], str]) -> str | Node:
    # print(root)
    if type(root) is Node:
        left = minimize_tree(root.left, nodedic)
        right = minimize_tree(root.right, nodedic)
        if (left, right) in nodedic:
            res = nodedic[left, right]
            return res
        else:
            nodedic[left, right] = root
            return root
    else:
        # print(type(root))
        if root not in nodedic:
            nodedic[root] = root
        return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Minimum SLP.")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument(
        "--log_level",
        type=str,
        help="log level, DEBUG/INFO/CRITICAL",
        default="CRITICAL",
    )
    args = parser.parse_args()
    if args.file == "" and args.str == "":
        parser.print_help()
        sys.exit()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.str != "":
        text = bytes(args.str, "utf-8")

    else:
        text = open(args.file, "rb").read()

    if args.log_level == "DEBUG":
        logger.setLevel(DEBUG)
    elif args.log_level == "INFO":
        logger.setLevel(INFO)
    elif args.log_level == "CRITICAL":
        logger.setLevel(CRITICAL)

    exp = AttractorExp.create()
    exp.algo = "slp-naive"
    exp.file_name = os.path.basename(args.file)
    exp.file_len = len(text)

    total_start = time.time()

    minsz = len(text) * 2
    ming = None
    solutioncounter = 0
    for tree in enum_ordered(text):
        solutioncounter += 1
        logger.info(tree)
        nodedic = {}
        rt = minimize_tree(tree, nodedic)
        sz = len(nodedic)
        if sz < minsz:
            minsz = sz
            ming = copy.deepcopy(nodedic)

    exp.time_total = time.time() - total_start
    exp.time_prep = exp.time_total
    exp.factor_size = minsz
    exp.sol_nvars = solutioncounter

    print("minimum SLP size = %s" % minsz)
    print("grammar: %s" % ming)

    if args.output == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(args.output, "w") as f:
            json.dump(exp, f, ensure_ascii=False)
