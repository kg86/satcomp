import argparse
import os
import subprocess
import sys
from typing import Optional

from bms import BiDirType, bd_info


def bms_naive(input_file: str, timeout: Optional[float] = None) -> BiDirType:
    input_file = os.path.abspath(input_file)
    cmd = f"cd rust && cargo run --bin optimal_bms -- --input_file {input_file}"
    print(cmd)
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=timeout)
    last1 = out.rfind(b"\n")
    last2 = out.rfind(b"\n", 0, last1)
    bd = eval(out[last2:last1])
    return BiDirType(bd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the smallest bidirectional macro scheme (BMS).")
    parser.add_argument("--file", type=str, help="input file", default="")

    args = parser.parse_args()
    if args.file == "":
        parser.print_help()
        sys.exit()
    return args


if __name__ == "__main__":
    args = parse_args()
    bd = bms_naive(args.file)
    text = open(args.file, "rb").read()
    print(bd_info(bd, text))
