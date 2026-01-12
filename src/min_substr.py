import argparse
import sys
from enum import Enum, auto
from typing import AnyStr, List, Tuple

import stralgo


class Mode(Enum):
    size_min_substr = auto()
    size_min_right_substr = auto()
    test = auto()
    exp = auto()


def main(input_file):
    print(input_file)
    with open(input_file, "r") as f:
        text = f.read()
    n = len(text)
    print(n)
    # run fast
    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    msubstr = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    print(len(msubstr))


def test():
    text = "ab"
    assert len(stralgo.minimum_substr(text)) == 2
    pos_lens = stralgo.minimum_substr(text)
    show(text, pos_lens)
    print()
    pos_lens = stralgo.minimum_right_substr(text)
    show(text, pos_lens)
    assert len(stralgo.minimum_right_substr(text)) == 2


def exp():
    files = [
        "data/calgary/bib",
        "data/calgary/book1",
        "data/calgary/book2",
        "data/calgary/geo",
        "data/calgary/news",
        "data/calgary/obj1",
        "data/calgary/obj2",
        "data/calgary/paper1",
        "data/calgary/paper2",
        "data/calgary/paper3",
        "data/calgary/paper4",
        "data/calgary/paper5",
        "data/calgary/paper6",
        "data/calgary/pic",
        "data/calgary/progc",
        "data/calgary/progl",
        "data/calgary/progp",
        "data/calgary/trans",
        "data/cantrbry/alice29.txt",
        "data/cantrbry/asyoulik.txt",
        "data/cantrbry/cp.html",
        "data/cantrbry/fields.c",
        "data/cantrbry/grammar.lsp",
        "data/cantrbry/kennedy.xls",
        "data/cantrbry/lcet10.txt",
        "data/cantrbry/plrabn12.txt",
        "data/cantrbry/ptt5",
        "data/cantrbry/sum",
        "data/cantrbry/xargs.1",
        "data/misc/abcd.txt",
        "data/misc/banana.txt",
        "data/misc/fib03.txt",
        "data/misc/fib04.txt",
        "data/misc/fib05.txt",
        "data/misc/fib06.txt",
        "data/misc/fib07.txt",
        "data/misc/fib08.txt",
        "data/misc/fib09.txt",
        "data/misc/fib10.txt",
        "data/misc/fib11.txt",
        "data/misc/fib12.txt",
        "data/misc/fib13.txt",
        "data/misc/hoge.txt",
        "data/misc/hoge0.txt",
        "data/misc/pds01.txt",
        "data/misc/pds02.txt",
        "data/misc/pds03.txt",
        "data/misc/pds04.txt",
        "data/misc/pds05.txt",
        "data/misc/pds06.txt",
        "data/misc/pds07.txt",
        "data/misc/thuemorse01.txt",
        "data/misc/thuemorse02.txt",
        "data/misc/thuemorse03.txt",
        "data/misc/thuemorse04.txt",
        "data/misc/thuemorse05.txt",
        "data/misc/thuemorse06.txt",
        "data/misc/thuemorse07.txt",
        "data/misc/trib05.txt",
        "data/misc/trib06.txt",
        "data/misc/trib07.txt",
        "data/misc/trib08.txt",
        "data/misc/trib09.txt",
    ]
    # files = files[-10:]
    algos = [stralgo.minimum_substr_sa, stralgo.minimum_right_substr_sa]
    print("file, len, nlrmin, nrmin, nlrmin/nrmin, total-lrmin, total-rmin, total-lrmin/total-rmin")
    for f in files:
        text = open(f, "rb").read()
        sa = stralgo.make_sa_MM(text)
        isa = stralgo.make_isa(sa)
        lcp = stralgo.make_lcpa_kasai(text, sa, isa)
        line = [f.split("/")[-1], len(text)]
        res = []
        for algo in algos:
            pos_len = algo(text, sa, isa, lcp)
            total_len = sum(l for _, l in pos_len)
            res.append(len(pos_len))
            res.append(total_len)
        line.extend([res[0], res[2], res[0] / res[2], res[1], res[3], res[1] / res[3]])
        # print(line)
        print(",".join(map(str, line)))


def show(text: AnyStr, substrs: List[Tuple[int, int]]):
    for i, l in substrs:
        print(text[i : i + l])


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum SLP.")
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--mode", type=str, help="[size_min_substr, size_min_right_substr]", default="")
    args = parser.parse_args()

    if args.mode == "size_min_substr":
        args.mode = Mode.size_min_substr
    elif args.mode == "size_min_right_substr":
        args.mode = Mode.size_min_right_substr
    elif args.mode == "test":
        args.mode = Mode.test
    elif args.mode == "exp":
        args.mode = Mode.exp
    else:
        parser.print_help()
        sys.exit()

    if args.mode not in [Mode.test, Mode.exp]:
        if args.file == "" and args.str == "":
            parser.print_help()
            sys.exit()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.mode in [Mode.test, Mode.exp]:
        if args.mode == Mode.test:
            test()
        elif args.mode == Mode.exp:
            exp()
    else:
        text = ""
        if args.str != "":
            text = bytes(args.str, "utf-8")
        elif args.file != "":
            text = open(args.file, "rb").read()

        if args.mode == Mode.size_min_substr:
            res = stralgo.minimum_substr(text)
        elif args.mode == Mode.size_min_right_substr:
            res = stralgo.minimum_right_substr(text)
        else:
            assert False
        total_length = sum(l for _, l in res)
        total = len(res)
        print(len(res), total_length)
