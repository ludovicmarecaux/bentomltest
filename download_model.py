import bentoml

from sklearn import svm
from sklearn import datasets

from sklearn.linear_model import SGDClassifier

# Load training data set
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train the model
clf = SGDClassifier()
clf.fit(X, y)

# Save model to the BentoML local Model Store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print("fini")