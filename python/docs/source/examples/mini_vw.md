# Mini VW



```python

import sys
from vowpalwabbit import pyvw

learnFromStrings = False

# this is a stupid program that basically mimics vw's behavior,
# mostly for the purpose of speed comparisons


def mini_vw(inputFile, numPasses, otherArgs):
    vw = pyvw.vw(otherArgs)
    for p in range(numPasses):
        print('pass', (p+1))
        h = open(inputFile, 'r')
        for l in h.readlines():
            if learnFromStrings:
                vw.learn(l)
            else:
                ex = vw.example(l)
                vw.learn(ex)
                vw.finish_example(ex)

        h.close()
    vw.finish()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: mini_vw.py (inputFile) (#passes) (...other VW args...)')
        exit()
    inputFile = sys.argv[1]
    numPasses = int(sys.argv[2])
    otherArgs = ' '.join(sys.argv[3:])

    mini_vw(inputFile, numPasses, otherArgs)
```