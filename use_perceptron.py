import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

print(df.tail())
print("\n")

# features and labels
print(df.iloc[145:150, 0:5], "\n")

# labels
y = df.iloc[0:100, 4].values
print(y, "\n")

# convert labels to integers
y = np.where(y == 'Iris-setosa', -1, 1)
print(y, "\n")


# extract features sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
print(X, "\n")

# Graph as points in 2D
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

"""
Looking at the graph we can clearly see the data is linearly separable.
This is good because perceptron will only converge if the data is
linearly separable.
"""

perceptron = Perceptron(0.1, 10)
perceptron.fit(X, y)
plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.show()

"""
Since the margin between the data is rather large we can get a perceptron
that has 100% accuracy. This is not always the case. If a perceptron cannot
get 100%. Use of SVM might be useful to explore
"""

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v' )
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_map = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=color_map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=color_map(idx),
                    marker=markers[idx], label=cl)


plot_decision_regions(X, y, classifier=perceptron)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()