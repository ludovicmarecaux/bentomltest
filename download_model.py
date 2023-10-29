import bentoml
import pandas as pd


from sklearn import svm
from sklearn import datasets

from sklearn.linear_model import SGDClassifier

# Load training data set
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_df=pd.DataFrame(X,columns=["sepal_length","sepal_width","petal_length","petal_width"])
y_df=pd.DataFrame(y,columns=["species"])
df=pd.merge(X_df, y_df, left_index=True, right_index=True)

df=df.drop_duplicates(keep="first")
num_column=["sepal_length","sepal_width","petal_length","petal_width"]

from sklearn.model_selection import train_test_split
X_tts=df[num_column]
y_tts=df.species
X_train,X_test,y_train,y_test=train_test_split(X_tts,y_tts,test_size=0.20,stratify=y_tts)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[num_column]=sc.fit_transform(X_train[num_column])
X_test[num_column]=sc.transform(X_test[num_column])

from sklearn.metrics import accuracy_score

def objective(trial):
    # Define hyperparameters
    c = trial.suggest_float("C", 1e-10, 1e10, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])

    # Create and train the model
    svm = SVC(C=c, kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

import optuna
from sklearn.svm import SVC
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(sampler=sampler, direction='maximize')
study.optimize(objective, n_trials=200)

clf=SVC()
clf.set_params(**study.best_params)


clf.fit(X, y)

# Save model to the BentoML local Model Store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print("fini")