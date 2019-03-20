import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn_porter import Porter

X = np.array([[1, 2],
              [2, 4],
              [3, 6],
              [4, 8],
              [5, 10],
              [6, 12]])

y = [0, 0, 0, 1, 1, 1]
clf = svm.SVC(kernel='linear', C=1.0, gamma=0.001)
clf.fit(X, y)

porter = Porter(clf)
output = porter.export()

file = open("LinearSVC.java", "w")
file.write(output)
file.close()


print(clf.predict([[10, 10]]))
print(clf.predict([[1, 1]]))
w = clf.coef_[0]
print(w)
