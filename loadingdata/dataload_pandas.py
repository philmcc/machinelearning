# Load CSV using Pandas
# Assumes that the datafile is in working dir
import pandas
filename = 'pima-indians-diabetes.data'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = pandas.read_csv(filename, names=names)
print(data.shape)