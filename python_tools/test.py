import numpy as np

# x = np.random.normal(0, 1, 16).reshape([4,4])

x = [(1,2),(3,4),(5,6)]
a, b = zip(*x)

print(a)
print(b)