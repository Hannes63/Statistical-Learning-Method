from NaiveBayes import *


X = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
     [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1,
     1, 1, 1, 1, -1]
model = NaiveBayes()
model.fit(X, y)
X0 = [2, 'S']
y0 = model.predict(X0)
print(y0)
