import numpy as np
import random
import time
from DecisionTree import DecisionTree


# age, has work, has house, credit level(0~4)
X = [['youth', 0, 0, 0], ['youth', 0, 0, 1], ['youth', 1, 0, 1], ['youth', 1, 1, 0], ['youth', 0, 0, 0],
     ['middle', 0, 0, 0], ['middle', 0, 0, 1], ['middle', 1, 1, 1], ['middle', 0, 1, 2], ['middle', 0, 1, 2],
     ['old', 0, 1, 2], ['old', 0, 1, 1], ['old', 1, 0, 1], ['old', 1, 0, 2], ['old', 0, 0, 0]]
y = [0, 0, 1, 1, 0,
     0, 0, 1, 1, 1,
     1, 1, 1, 1, 0]

d = DecisionTree()
root = d.fit(X, y)
X0 = ['old', 0, 1, 2]
print(d.predict(X0))

