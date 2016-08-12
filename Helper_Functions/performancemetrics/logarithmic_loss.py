# Logarithmic loss (or logloss) is a performance metric for evaluating the predictions of probabilities 
# of membership to a given class. The scalar probability between 0 and 1 can be seen as a measure of 
# confidence for a prediction by an algorithm. Predictions that are correct or incorrect are rewarded or punished 
# proportionally to the confidence of the prediction. 

# Cross Validation Classification LogLoss
import pandas
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
num_instances = len(X)
seed = 7
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed) 
model = LogisticRegression()
scoring = 'log_loss'
results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())