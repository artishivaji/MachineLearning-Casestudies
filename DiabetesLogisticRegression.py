import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from warning import simplefilter

print("Marvellous Infosystems by Piyush khairnar")
print("Diabetes predictor using Logistic Regression")

diabetes = pd.read_csv('diabetes.csv')

print("Column of Dataset")
print(diabetes.columns)

print("First 5 records of dataset")
print(diabetes.head())

print("Dimension of diabetes data:{}".format(diabetes.shape))

X_train,X_test,y_train,y_test = train_test_split(diabete.loc[:,diabetes.columns!= 'Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

logreg= LogisticRegression().fit(X_train,y_train)

print("Accuracy on training set:{:.3f}".format(tree.score(X_train,y_train)))

print("Accuracy on test set:{:.3f}".format(tree.score(X_test,y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)
print("Accuracy on training set:{:.3f}".format(tree.score(X_train,y_train)))
print("Accuracy on test set:{:.3f}".format(tree.score(X_test,y_test)))














