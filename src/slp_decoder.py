import sys, json
import array
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether <stdin> JSON contains valid BMS")
    parser.add_argument("--file", type=str, help="original text input file", default="")
    parser.add_argument("--json", type=str, help="JSON file to parse", default="")
    args = parser.parse_args()

    if args.json: 
        data = json.load(open(args.json)) 
    else:
        data = json.load(sys.stdin) 
    
    grammar = eval(data["factors"])

    root = grammar[0]
    text = [None for _ in range(root[1])]
    slp = grammar[1]


    stack = [root]
    cur_textpos = 0
    while len(stack) > 0:
        node = stack[-1]
        # print(f"Visit node {node}")
        if slp[node] != None:
            stack.append(slp[node][0])
            continue
        elif node[1]-node[0] > 1:
            assert node[2] != None
            refnode = (node[2],node[2]+node[1]-node[0],None)
            assert refnode in slp
            stack.append(slp[refnode][0])
            continue
        else:
            assert node[2] != None
            textpos = cur_textpos
            assert text[textpos] == None, f"text[{textpos}] = {text[textpos]} already set"
            text[textpos] = node[2]
            # print(f"T[{textpos}] <- {node[2]}")
            cur_textpos += 1
        stack.pop()
        # move to lowest ancestor from which we are not its right child
        while len(stack) > 0:
            top = stack[-1]
            if slp[top] == None:
                refnode = (top[2],top[2]+top[1]-top[0],None)
                assert refnode in slp
                if slp[refnode][1] != node:
                    break
            elif slp[top][1] != node:
                break
            node = stack.pop()
        if len(stack) > 0:
            top = stack[-1]
            leftchildsize = 0
            if slp[top] == None:
                refnode = (top[2],top[2]+top[1]-top[0],None)
                stack.append(slp[refnode][1])
                leftchildsize = slp[refnode][0][1]-slp[refnode][0][0]
            else:
                stack.append(slp[top][1])
                leftchildsize = slp[top][0][1]-slp[top][0][0]

    is_correct=True
   

    output = ""
    for i in range(len(text)):
        c = text[i]
        if isinstance(c, int):
            output += chr(c)
        else:
            is_correct = False
            print(f"could not recover text position {i}")
            output += '?'
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
        sys.exit("not a valid grammar for the input")

