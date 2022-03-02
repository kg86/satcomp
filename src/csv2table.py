import csv
import sys

table_template = """
\\begin{{tabular}}{{{}}}
\\hline
{}
\\\\ \\hline
\\end{{tabular}}
"""


def main(fname: str, vtype: str):
    assert vtype == "str" or vtype == "float"
    res = []
    # header = ''
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        # header.pop(1)
        header[1] = "len"
        header[2] = "lz77"
        header[-2] = "SA"
        header[-1] = "BMS"
        res.append(" & ".join(header))
        for row in reader:
            row[0] = row[0].split("-")[0]
            row[0] = row[0].split(".txt")[0]
            # row.pop(1)
            line = [row[0], row[1]]
            for elm in row[2:]:
                if elm.startswith("timeout"):
                    line.append("T")
                elif elm.startswith("error"):
                    line.append("M")
                else:
                    try:
                        if vtype == "float":
                            line.append("{:.4f}".format(float(elm)))
                        else:
                            line.append(elm)
                    except:
                        line.append(elm)

            res.append(" & ".join(line))
    table_format = "|{}|".format("|".join(["c" for _ in header]))
    table_body = " \\\\ \\hline\n".join(res)
    print(table_template.format(table_format, table_body))


def size_table(fname):
    res = []
    # header = ''
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        header.pop(1)
        header[1] = "lz77"
        header[-2] = "SA"
        header[-1] = "BMS"
        res.append(" & ".join(header))
        for row in reader:
            row[0] = row[0].split("-")[0]
            row.pop(1)
            line = []
            for elm in row:
                if elm.startswith("timeout"):
                    line.append("T")
                elif elm.startswith("error"):
                    line.append("M")
                else:
                    line.append(elm)
            res.append(" & ".join(line))
    table_format = "|{}|".format("|".join(["c" for _ in header]))
    table_body = " \\\\ \\hline\n".join(res)
    print(table_template.format(table_format, table_body))


if __name__ == "__main__":
    fname = sys.argv[1]
    vtype = sys.argv[2]
    # size_table(fname)
    main(fname, vtype)
