import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
def load_data(data_dir):
    df = pd.read_csv(data_dir+"train.csv")
    y = df.label
    df.drop('label', axis=1, inplace=True)
    df = df.values
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    return X_train, y_train, X_test, y_test

def mapper(inRow):
    reShaped = inRow.reshape(28, 28)
    rsltRow=[]
    for y in range(0, 28):
        tmpSum = 0;
        for x in range(0, 28):
            if(reShaped[y][x]!=0):
                tmpSum +=1
        # rsltRow.append([y,tmpSum])
        rsltRow.append(tmpSum)
    rsltCol=[]
    for x in range(0, 28):
        tmpSum = 0;
        for y in range(0, 28):
            if(reShaped[y][x]!=0):
                tmpSum +=1
        # rsltCol.append([x,tmpSum])
        rsltCol.append(tmpSum)
    return [rsltRow, rsltCol]

def dataToSeries(dataset):
    rowArray = []
    # colArray = []
    for i in range(0, len(dataset)):
        row = mapper(dataset[i])
        rowArray.append(row)
        # colArray.append(col)
    return to_time_series(rowArray)

X_train, y_train, X_test, y_test = load_data('data/')
X_train_ts =  dataToSeries(X_train)
X_test_ts = dataToSeries(X_test)

knn_clf_dtw = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
knn_clf_dtw.fit(X_train_ts, y_train)
predicted_labels_dtw = knn_clf_dtw.predict(X_test_ts)
print("knn with dtw: \n", accuracy_score(y_test, predicted_labels_dtw))
print("Classification report: \n", classification_report(y_test, predicted_labels_dtw))
print("Confusion matrix: \n", confusion_matrix(y_test, predicted_labels_dtw))


unlabaled =   df = pd.read_csv("data/test.csv")
unlabaled = unlabaled.values
unlabaled_ts = dataToSeries(unlabaled)
plt.imshow(unlabaled[165].reshape((28, 28)))
predicted_label_dtw = knn_clf_dtw.predict(unlabaled_ts)