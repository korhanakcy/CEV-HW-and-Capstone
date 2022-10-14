import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# GaussianNB with different var_smoothing values
var_smoothing = np.logspace(0,-9, num=20)
accuracy = []
for i in var_smoothing:
    gnb = GaussianNB(var_smoothing=i)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))

plt.plot(var_smoothing, accuracy)
plt.xlabel("var_smoothing")
plt.ylabel("accuracy")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# DecisionTreeClassifier with different max_depth values
max_depth = np.arange(1, 10)
accuracy = []
for i in max_depth:
    dtc = DecisionTreeClassifier(max_depth=i)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))

plt.plot(max_depth, accuracy)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.show()
