import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def MarvellousPlayPredictor(data_path):

    #Step 1 : Load data
    data = pd.read_csv(data_path,index_col=0)

    print("Size of Actual dataset",len(data))

    #Step 2 : Clean, Prepare and manipulate data
    feature_names = ['Whether',Temperature]

    print("Names of Features",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.play

    #creating labelEncoder
    le = preprocessing.LabelEncoder()

    #Converting string labels into numbers.
    weather_encoded=le.fit_transform(whether)
    print(weather_encoded)

    #converting string labels into numbers
    temp_encoded=le.fit_transform(Temperature)
    label=le.fit_transform(play)

    print(temp_encoded)

    #combining weather and temp into single listof tuples
    features=list(zip(weather_encoded,temp_encoded))

    #step 3 : Train Data
    model = KNeighborsClassifier(n_neighbors=3)

    #Train the model using the training sers
    model.fit(features,label)

    # step4 : Test data
    predicted= model.predict([[0,2]])

    print(predicted)

def main():
    print("Marvellous Infosystem by piyush khairnar")

    print("Machine Learning Application")
    print("Play predictor application using K Nearest Knighbor algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()








