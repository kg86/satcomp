import stralgo


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


if __name__ == "__main__":
    input_file = "cantrbry/grammar.lsp"
    # input_file = 'fuga.txt'
    main(input_file)
