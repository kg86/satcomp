import sys, json
import array
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether <stdin> JSON contains valid BMS")
    parser.add_argument("--file", type=str, help="original text input file", default="")
    parser.add_argument("--json", type=str, help="JSON input file", default="")
    args = parser.parse_args()

    if(args.json):
        data = json.load(open(args.json, "r"))
    else:
        data = json.load(sys.stdin) 
    text = array.array('b',[])
    recovered = []
    for factor in data["output"]:
        if factor[0] == -1:
            text.append(factor[1])
            recovered.append(True)
        else:
            text.extend([0 for _ in range(factor[1])])
            recovered.extend([False for _ in range(factor[1])])
    pointers = [ [] for _ in range(len(text))]

    assert len(pointers) == len(text)
    assert len(recovered) == len(text)

    requests = set()
    p = 0
    for factor in data["output"]:
        if factor[0] == -1:
            p+=1
            continue
        if factor[0] != -1:
            ref = factor[0]
            for j in range(factor[1]):
                assert recovered[p+j] == False
                pointers[ref+j].append(p+j)
                requests.add(ref+j)
            p+=factor[1]

    while len(requests) > 0:
        newrequests = set()
        for request in requests:
            if recovered[request] == False: 
                newrequests.add(request)
                continue
            for pos in pointers[request]: 
                assert recovered[pos] == False
                text[pos] = text[request]
                recovered[pos] = True
        if len(requests) == len(newrequests):
            print(f"cycle detected!")
            sys.exit(2)

        requests = newrequests

    is_correct=True

    output = ""
    for i in range(len(text)):
        output += chr(text[i])
        if recovered[i] == False:
            print(f"could not recover text position {i}")
            is_correct = False
    print(output)
    


    if args.file:
        with open(args.file, "r") as f:
            read = f.read()
            if len(read) != len(output):
                print(f"decoded file sizes mismatch: original-length:{len(read)} vs. decoded-length:{len(output)}")
                is_correct=False
            for i in range(min(len(read),len(output))):
                if read[i] != output[i]:
                    print(f"mismatch at position {i} : decoded={output[i]} original={read[i]}")
                    is_correct=False
    if is_correct:
        print("output correct!")
    else:
        sys.exit("not a valid BMS for the input")

