from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def MarvellousSVM():
    #load dataset
    cancer = datasets.load_breast_cancer()

    #print the names of the 13 features
    print("Features of the cancer dataset:",cancer.feature_names)

    #print the label type of cancer(malignant'benign')
    print("Labels of the cancer dataset:",cancer.target_names)

    #print data(feature)shape
    print("Shape of dataset is:",cancer.data.shape)

    #print the cancer data features(top5 recors)
    print("First 5 records are:")
    print(cancer.data[0:5])

    #print thecancer labels(0:malignant,1:benign)
    print("Target of dataset:",cancer.target)

    #Split dataset into training set and test set
    X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear')

    #Train the model using the training sets
    clf.fit(X_train,y_train)

    #Predict the responses for test dataset
    y_pred = clf.predict(X_test)

    #Model accuracy:how often is the classifier correct?
    print("Accuracy of the model is:",metrics.accuracy_score(y_test,y_pred)*100)

def main():
    print("Marvellous support vector machine")

    MarvellousSVM()

if __name__ == "__main__":
    main()













