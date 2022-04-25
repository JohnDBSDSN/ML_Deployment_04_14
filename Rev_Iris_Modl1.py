#Importing libraries for data manipulation
import pandas as pd
import numpy as np

#Importing libraries for visualization 
#import matplotlib.pyplot as plt
#import seaborn as sns

#Import libraries for ignoring warning
import warnings
warnings.filterwarnings('ignore')

#Importing library for label encoding
from sklearn.preprocessing import LabelEncoder

#Importing library for splitting data into train and test
from sklearn.model_selection import train_test_split

#Importing libraries for model training
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
#Importing library for parameter tuning
from sklearn.model_selection import GridSearchCV
#Loading data
import joblib


url = "D:\Data Analytics - Learnbay\Deployment\ML_Deployment\Iris.csv"
names = ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']

data = pd.read_csv(url , names=names)
print('Data type:', type(data))
print('Data id type:\n', type(data['Id']))
print('Data id Value:\n', data['Id'])
#x['Cost'] = pd.to_numeric(x['Cost'].str.replace(",", ""), errors='coerce')
#data['Id'] = pd.to_numeric(data['Id'].str.replace(",", ""), errors='ignore')
#data['SepalLengthCm'] = pd.to_numeric(data['SepalLengthCm'].str.replace(".", ""), errors='ignore')
#data['SepalWidthCm'] = pd.to_numeric(data['SepalWidthCm'].str.replace(".", ""), errors='ignore')
#data['PetalLengthCm'] = pd.to_numeric(data['PetalLengthCm'].str.replace(".", ""), errors='ignore')
#data['PetalWidthCm'] = pd.to_numeric(data['PetalWidthCm'].str.replace(".", ""), errors='ignore')
#data['Species'] = pd.to_numeric(data['Species'].str.replace(".", ""), errors='ignore')

#data['Id'] = data['Id'].astype(float, errors = 'raise')
#data['SepalLengthCm'] = data['SepalLengthCm'].astype(float, errors = 'raise')
#data['SepalWidthCm'] = data['SepalWidthCm'].astype(float, errors = 'raise')
#data['PetalLengthCm'] = data['PetalLengthCm'].astype(float, errors = 'raise')
#data['PetalWidthCm'] = data['PetalWidthCm'].astype(float, errors = 'raise')
#data['Species'] = data['Species'].astype(float, errors = 'raise')

#data['Id'] = pd.to_numeric(data['Id'])
#data['SepalLengthCm'] = pd.to_numeric(data['SepalLengthCm'])
#data['SepalWidthCm'] = pd.to_numeric(data['SepalWidthCm'])
#data['PetalLengthCm'] = pd.to_numeric(data['PetalLengthCm'])
#data['PetalWidthCm'] = pd.to_numeric(data['PetalWidthCm'])
#data['Species'] = pd.to_numeric(data['Species'])




data = data.drop('Id', axis=1)

print(data)

print(data.info())
print('Describe:\n\n',data.describe())

#Checking value count of the data
for i in data.columns:
  print(i,'=>\n')
  print(data[i].value_counts())
  print('O1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~O1\n')

  #Checking unique values in the data
for i in data.columns:
  print(i,'=>\n')
  print(set(data[i]))
  print('O2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~O2\n')

  #Checking missing values in the data
print('Missing Values in Data:', data.isnull().sum())

#Creating correlation matrix
print('Correlation Matrix:', data.corr())

#Implementing label encoding on species variable
from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#data['Species'] = le.fit_transform(data['Species'])
#print('After transformation:', data['Species'])

#Checking the data
print('Checking Data:', data.head())

# Splitting The Data into Training And Testing Dataset

train, test = train_test_split(data, test_size = 0.3)# in this our main data is split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
print(train.shape)
print(test.shape)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features
train_y=train.Species# output of our training data
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
test_y =test.Species   #output value of test data

print('train_X:\n\n', (train_X).head())
print('train_y:\n\n', (train_y).head())
print('test_X:\n\n', (test_X).head())
print('test_y:\n\n', (test_y).head())

#print(train_y.head())

#Decision Tree
model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

#Save the model
#joblib.dump(model, 'D:\Data Analytics - Learnbay\Deployment\Saving_Model\Joblib\Rev_Iris1.pkl')
joblib.dump(model, 'Rev_Iris_Modl.pkl')









