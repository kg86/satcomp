from slp_decoder import decode_slp
import satcomp.io as io
import sys

def verify_slp(read, grammar):
    output = decode_slp(grammar)

    is_correct = True
    if len(read) != len(output):
        print(f"decoded file sizes mismatch: original-length:{len(read)} vs. decoded-length:{len(output)}")
        is_correct=False
    for i in range(min(len(read),len(output))):
        if read[i] != output[i]:
            print(f"mismatch at position {i} : decoded={output[i]} original={read[i]}")
            is_correct=False
    return is_correct


if __name__ == "__main__":
    is_correct = io.verify_functor(verify_slp, 'verifies an SLP')
    if is_correct:
        print("output correct!")
    else:
        sys.exit("not a valid grammar for the input")
