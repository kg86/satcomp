import satcomp.stralgo as stralgo
import argparse
import sys

def min_substrings(text: bytes):

    yield f'#const textlength={len(text)}.'
    yield '% cover(start,end,position)'
    sa = stralgo.make_sa_MM(text)
    isa = stralgo.make_isa(sa)
    lcp = stralgo.make_lcpa_kasai(text, sa, isa)
    min_substrs = stralgo.minimum_substr_sa(text, sa, isa, lcp)
    for (b, l) in min_substrs:
        lcp_range = stralgo.get_lcprange(lcp, isa[b], l)
        occs = [sa[i] for i in range(lcp_range[0], lcp_range[1] + 1)]
        for position in set(occ + i + 1 for occ in occs for i in range(l)):
            yield f'cover({b+1},{b+l},{position}).'



def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum String Attractors.")
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
    if (
        (args.file == "" and args.str == "")
        or (args.log_level not in ["DEBUG", "INFO", "CRITICAL"])
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
    output = sys.stdout
    if args.output:
        output = open(args.output, "w", encoding='utf-8')

    for line in min_substrings(text):
        print(line, file=output)
