from slp_decoder import decode_slp
import satcomp.base as io
import sys

def verify_slp(text, output):
    output = decode_slp(output)

    is_correct = True
    if len(text) != len(output):
        print(f"decoded file sizes mismatch: original-length:{len(text)} vs. decoded-length:{len(output)}")
        is_correct=False
    for i in range(min(len(text),len(output))):
        if text[i] != output[i]:
            print(f"mismatch at position {i} : decoded={output[i]} original={text[i]}")
            is_correct=False
    return is_correct


if __name__ == "__main__":
    error_code = io.verify_functor(verify_slp, 'verifies an SLP')
    if error_code == 0:
        print("output correct!")
    else:
        print("not a valid grammar for the input")
        sys.exit(1)
