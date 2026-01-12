"""Generate the period-doubling sequence and write it to files."""

char0 = "a"
char1 = "b"


def pds(text: str) -> str:
    """Apply one period-doubling morphism step to `text`."""
    n = len(text)
    res = ""
    for i in range(n):
        if text[i] == char0:
            res += char0 + char1
        elif text[i] == char1:
            res += char0 + char0
        else:
            assert False
    return res


def make_dataset():
    """Write a small sequence of period-doubling strings to `data/misc/`."""
    file_template = "data/misc/pds{}.txt"
    num = 8
    res = ["" for _ in range(num)]
    res[0] = char0
    for i in range(1, num):
        file = file_template.format(("0" if i < 10 else "") + str(i))
        res[i] = pds(res[i - 1])
        with open(file, "w") as f:
            f.write(res[i])


if __name__ == "__main__":
    make_dataset()
