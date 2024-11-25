import argparse
import logging
import enum
from typing import Any
from satcomp.solver import MaxSatStrategy, SolverType

class LogLevel(enum.IntEnum):
    CRITICAL = logging.CRITICAL,
    ERROR   = logging.ERROR,
    WARNING = logging.WARNING,
    INFO    = logging.INFO,
    DEBUG   = logging.DEBUG,
    NOTSET  = logging.NOTSET,
    def __str__(self):
        return str(self.name)

def dump_wcnf_and_exit(wcnf, dumpfilename):
    if dumpfilename:
        wcnf.to_file(dumpfilename, compress_with='lzma')
        sys.exit(0)



def solver_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--file", type=str, help="input file", default="")
    parser.add_argument("--str", type=str, help="input string", default="")
    parser.add_argument("--prefix", type=int, help="parse only a prefix of input", default=0)
    parser.add_argument("--output", type=str, help="output file", default="")
    parser.add_argument("--dump", type=str, help="dump WCNF to <DUMP> instead of solving", default="")
    parser.add_argument(
        "--loglevel",
		type=lambda x: LogLevel[x],
		choices=list(LogLevel),
        help="log level",
        default="CRITICAL",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        help="verbosity level of the sat solver",
        default=0,
    )
    parser.add_argument(
        "--strategy",
        type=lambda x: MaxSatStrategy[x],
        choices=list(MaxSatStrategy),
        help="maxsat-solver strategy",
        default="LSU",
    )
    parser.add_argument(
        "--solver",
        type=lambda x: SolverType[x],
        choices=list(SolverType),
        help="sat-solver",
        default=SolverType.Glucose4,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="number of seconds spend for the SAT solver until a (maybe unfinished) solution must be reported",
        default=0,
    )
    return parser

def verify_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--file", type=str, help="original input file", default="")
    parser.add_argument("--str", type=str, help="original input string", default="")
    parser.add_argument("--prefix", type=int, help="parse only a prefix of the original input", default=0)
    parser.add_argument("--json", type=str, help="JSON file or <STDIN>", default="")
    parser.add_argument(
        "--loglevel",
		type=lambda x: LogLevel[x],
		choices=list(LogLevel),
        help="log level",
        default="CRITICAL",
    )
    return parser

def decode_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--json", type=str, help="JSON file or <STDIN>", default="")
    parser.add_argument("--output", type=str, help="output file or <STDOUT>", default="")
    parser.add_argument(
        "--loglevel",
		type=lambda x: LogLevel[x],
		choices=list(LogLevel),
        help="log level",
        default="CRITICAL",
    )
    return parser


import sys

def read_input(args, use_stdin : bool = True) -> bytes:
    """ use_stdin: read from stdin in case that no file is specified in args """
    text = ''
    if args.str != "":
        text = args.str.encode("utf8") if args.prefix == 0 else args.str.encode("utf8")[:args.prefix]
    elif args.file != "":
        with open(args.file, "rb") as f:
            text = f.read() if args.prefix == 0 else f.read(args.prefix)
    elif use_stdin:
        text = sys.stdin.read() if args.prefix == 0 else sys.stdin.read(args.prefix)
    return text

import json
from typing import Any

def read_json(args, use_stdin : bool = True):
    """ use_stdin: read from stdin in case that no file is specified in args """
    data = None
    if args.json != "":
        data = json.load(open(args.json, "r"))
    elif use_stdin:
        data = json.load(sys.stdin)
    return data


def write_json(outfilename : str, exp : Any):
    if outfilename == "":
        print(exp.to_json(ensure_ascii=False))  # type: ignore
    else:
        with open(outfilename, "w") as f:
            f.write(exp.to_json(ensure_ascii=False))
            # json.dump(exp, f, ensure_ascii=False)

def decode_functor(decoder, description : str):
    parser = decode_parser(description)
    args = parser.parse_args()
    data = read_json(args, use_stdin=True)
    if data == None:
        return 2
    decodedtext = decoder(output = data["output"])
    if args.output:
        with open(args.output, "wb") as ostream:
            ostream.write(decodedtext)
    else:
        sys.stdout.write(decodedtext)


def verify_functor(verificator, description : str):
    parser = verify_parser(description)
    args = parser.parse_args()
    text = read_input(args, use_stdin=False)
    data = read_json(args, use_stdin=True)
    if data == None:
        return 2
    else:
        return 0 if verificator(text = text, output = data["output"]) else 1

# vim:fenc=utf-8 ff=unix ft=python ts=4 sw=4 sts=4 si et :

