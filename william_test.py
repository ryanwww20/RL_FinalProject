import numpy as np
from gymnasium import spaces

action_space = spaces.MultiBinary(n=15)
print(action_space.sample().reshape(-1, 1))
# reshape(-1, 1) means to convert the array to a 15x1 array
