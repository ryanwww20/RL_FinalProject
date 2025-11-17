import numpy as np


for i in range(15):
    layer = np.array([[(j+i) & 1 for j in range(15)]])
    print(layer)
