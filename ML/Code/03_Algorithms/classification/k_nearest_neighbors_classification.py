# KNN Classification
import pandas
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
num_instances = len(X)
random_state = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=random_state)
model = KNeighborsClassifier()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
