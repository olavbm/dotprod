import math
import numpy as np

def numpy_dot(a=[1,2,3], b=[11,22,33]):
    res = np.dot(a,b)
    return res

def self_made_dot(a=[1,2,3], b=[11,22,33]):
    if len(a) != len(b):
        raise TypeError

    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]

    return s

print(numpy_dot())
print(self_made_dot())

def test_dotprods():
    for x in range(10, 10000, 7):
        for y in range(0, 1000, 3):
            y = y / 3
            xs = [x + (y+i) for i in range(math.ceil(y))]
            ys = [y + (y+i) for i in range(math.ceil(y))]

            np = numpy_dot(xs, ys)
            sm = self_made_dot(xs, ys)
            if np != sm:
                print(math.fabs(np-sm))
                print(xs, ys)
                return False
    return True

print(test_dotprods())
