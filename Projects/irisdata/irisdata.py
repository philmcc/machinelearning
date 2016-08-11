# Iris Data Project


######
# 1. Prepare Problem
######
# a) Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# b) Load dataset
url = "iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
dataset = pandas.read_csv(url, names=names)

######
# 2. Summarize Data
######
# a) Descriptive statistics
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())


# b) Data visualizations
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
#plt.show()
# histograms
#dataset.hist()
#plt.show()
# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

######
# 3. Prepare Data
######
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

######
# 4. Evaluate Algorithms
######
# a) Split-out validation dataset
array = dataset.values
X = array[:,0:4] 	# inputs
Y = array[:,4]		# outpus
validation_size = 0.20	# Hold back 20% of data for later validation
seed = 7	# Random Seed for reproducability
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y,
    test_size=validation_size, random_state=seed) # Do the actual split!

# b) Test options and evaluation metric
num_folds = 10 # for k-fold cross validation (or 10 fold in this case)
num_instances = len(X_train) 
seed = 7 # Setting the seed so that each different model gets exactly the same data
scoring = 'accuracy' # Metric to evaluate the tests by - correct instances / total instances

# c) Spot-Check Algorithms
# Quickly test a number of models to get an idea of the best performing ones for this problem
models = []
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))
# evaluate each model in turn
# for each model return the mean of the accuracy and the standard deviation 
results = []
names = []
for name, model in models:
  kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
  cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold,
      scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

# d) Compare Algorithms
# Visualize those results
fig = plt.figure() 
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names)
#plt.show()

######
# 5. Improve Accuracy
######
# This is a simple problem and accuracy is already high so there is no need for this step in this case.
# a) Algorithm Tuning
# b) Ensembles

######
# 6. Finalize Model
######
# a) Predictions on validation dataset
# KNN looded good so move forward with that
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Looking back SVM had a higher accuracy so trying that
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# SVM loks great with a 94% accuracy score!

# b) Create standalone model on entire training dataset
# c) Save model for later use




