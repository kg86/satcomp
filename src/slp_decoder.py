from satcomp.measure import SLPType
import satcomp.io as io

#TODO: grammar could derive from SLPType, but then pyright gets crazy
# def decode_slp(grammar : SLPType) -> str:

def decode_slp(output) -> list[int]:
    grammar = eval(output)
    root = grammar[0]
    print(root)
    text = [0 for _ in range(root[1])]
    assigned = [False for _ in range(root[1])]
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
            assert not assigned[cur_textpos], f"text[{cur_textpos}] = {text[cur_textpos]} already set"
            text[cur_textpos] = node[2]
            assigned[cur_textpos] = True
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
            if slp[top] == None:
                refnode = (top[2],top[2]+top[1]-top[0],None)
                stack.append(slp[refnode][1])
            else:
                stack.append(slp[top][1])

    # is_correct=True
   
    for i in range(len(text)):
        if not assigned[i]:
            print(f"could not recover text position {i}")

    # output = []
    # for i in range(len(text)):
    #     if not assigned[i]:
    #         is_correct = False
    #         print(f"could not recover text position {i}")
    #         output.append('?')
    #     else:
    #         output.append(chr(text[i]))
    # if not is_correct:
    #     return ""
    return text


if __name__ == "__main__":
    io.decode_functor(lambda output: ''.join(map(lambda x: chr(x), decode_slp(output))) , 'decode an SLP')

