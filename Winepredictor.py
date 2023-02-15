from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    #Load dataset
    wine = datasets.load_wine()

    #print the names of the features
    print(wine.feature_names)

    #print the label species(class_0,class_1,class_2)
    print(wine.target_names)

    #print the wine data(top 5 recors)
    print(wine.data[0:5])

    #print the wine labels(0:Class_0,1:Class_1,2:Class_3)
    print(wine.target)

    #split dataset into training set and test set
    X_train,X_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3)#70% training and30%test

    #Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    #Train the model using the training sets
    knn.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)

    #Model Accuracy,how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

def main():
    print("Marvellous infosystem by piyush khairnar")
    print("Machine Learning Application")

    print("Wine predictor application using K Nearest Knighbor algorithm")

    WinePredictor()

if __name__ =="__main__":
    main()









