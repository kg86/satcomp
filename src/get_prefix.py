# trim the specified length prefix of given file, and save it.

import glob
import os


def make_dataset():
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


def main(in_file, out_file, pref_len: int):
    text = open(in_file, "rb").read()
    open(out_file, "wb").write(text[:pref_len])


if __name__ == "__main__":
    # main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    make_dataset()
