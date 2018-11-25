from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import recall_score

iris = datasets.load_iris()
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5, return_train_score=False)
sorted(scores.keys())

print(scores['test_recall_macro'])