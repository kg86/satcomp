import copy
import argparse
import os
import sys
import json
import time

import satcomp.io as io
from satcomp.measure import SLPType, SLPExp

from logging import CRITICAL, getLogger, DEBUG, INFO, StreamHandler, Formatter

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
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return "(%s %s)" % (self.left, self.right)


def enum_ordered(labels):
    if len(labels) == 1:
        yield labels[0]
    else:
        for i in range(1, len(labels)):
            for left in enum_ordered(labels[:i]):
                for right in enum_ordered(labels[i:]):
                    yield Node(left, right)


##############################################################################################


def minimize_tree(root, nodedic):
    # print(root)
    if type(root) == Node:
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


""" TODO: the output is not in the format as slp_solver! """

if __name__ == "__main__":
    parser = io.solver_parser('compute a minimum straight line program')
    args = parser.parse_args()
    logger.setLevel(int(args.loglevel))
    text = io.read_input(args)

    exp = SLPExp.create()
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
    exp.output_size = minsz
    exp.output = str(ming)
    exp.sol_nvars = solutioncounter

    # print("minimum SLP size = %s" % minsz)
    # print("grammar: %s" % ming)


    io.write_json(args.output, exp)
