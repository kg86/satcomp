import copy
import argparse
import os
import sys

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


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum SLP.")
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
        text = bytes(args.str, "utf-8")

    else:
        text = open(args.file, "rb").read()

    minsz = len(text) * 2
    ming = None
    for tree in enum_ordered(text):
        print(tree)
        nodedic = {}
        rt = minimize_tree(tree, nodedic)
        sz = len(nodedic)
        if sz < minsz:
            minsz = sz
            ming = copy.deepcopy(nodedic)
    print("minimum SLP size = %s" % minsz)
    print("grammar: %s" % ming)
