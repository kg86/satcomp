from bidirectional import BiDirType, bd_info
import subprocess
import argparse
import sys
import os


def bidirectional_naive(input_file: str, timeout: float = None) -> BiDirType:
    input_file = os.path.abspath(input_file)
    cmd = f"cd rustr-master && cargo run --bin optimal_bms -- --input_file {input_file}"
    print(cmd)
    out = subprocess.check_output(
        cmd, shell=True, stderr=subprocess.DEVNULL, timeout=timeout
    )
    last1 = out.rfind(b"\n")
    last2 = out.rfind(b"\n", 0, last1)
    bd = eval(out[last2:last1])
    return BiDirType(bd)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Minimum Bidirectional Scheme")
    parser.add_argument("--file", type=str, help="input file", default="")
    # parser.add_argument("--output", type=str, help="output file", default="")

    args = parser.parse_args()
    if args.file == "":
        parser.print_help()
        sys.exit()
    return args


if __name__ == "__main__":
    args = parse_args()
    bd = bidirectional_naive(args.file)
    text = open(args.file, "rb").read()
    print(bd_info(bd, text))
