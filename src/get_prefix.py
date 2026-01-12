"""Trim prefixes of input files to create prefix datasets."""

import glob
import os


def make_dataset():
    """Create prefix-trimmed datasets for a fixed set of input corpora."""
    dirs = ["data/calgary", "data/cantrbry"]
    prefs = range(50, 500, 50)
    prefs = [10000]
    for dir in dirs:
        files = glob.glob(dir + "/*")
        for file in files:
            for pref in prefs:
                out_dir = dir + "_pref"
                out_file = os.path.basename(file) + "-" + str(pref)
                main(file, os.path.join(out_dir, out_file), pref)


def main(in_file: str, out_file: str, pref_len: int):
    text = open(in_file, "rb").read()
    open(out_file, "wb").write(text[:pref_len])


if __name__ == "__main__":
    make_dataset()
