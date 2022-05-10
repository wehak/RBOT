import numpy as np

# x = np.random.normal(0, 1, 16).reshape([4,4])

x = np.identity(4)
y = np.zeros(4*4*3).reshape([4,4,3])

print(x.any())
print(y.any())