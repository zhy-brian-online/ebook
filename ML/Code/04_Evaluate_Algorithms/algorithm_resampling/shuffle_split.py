# Evaluate using Shuffle Split Cross Validation
import pandas
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_samples = 10
test_size = 0.33
num_instances = len(X)
seed = 7
kfold = cross_validation.ShuffleSplit(n=num_instances, n_iter=num_samples, test_size=test_size, random_state=seed)
model = LogisticRegression()
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)