from perceptron import *


X = np.array([[3, 3], [4, 3], [1, 1], [2, 2], [2, 1]])
y = [1, 1, -1, 1, -1]
p = Perceptron(learning_rate=1.0, plot='last')
p.fit(X, y)
X0 = [[2, 4], [6, 6], [3, 1]]
y = p.predict(X0)
print(y)
