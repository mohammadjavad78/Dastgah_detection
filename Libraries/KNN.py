from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

class KNN:
    def __init__(self,X,Y,testsize=1/3,inp=True,X_train="None",X_test="None",y_train="None",y_test="None"):
        self.X=X
        self.Y=Y
        if(inp):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y, test_size=testsize, random_state=0, stratify=Y)
        else:
            self.X_train=X_train
            self.X_test=X_test
            self.y_train=y_train
            self.y_test=y_test


    def change(self,X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

    def get(self):
        return self.X_train,self.X_test,self.y_train,self.y_test

    def accurate(self,K=1):
        scaler = preprocessing.StandardScaler()
        X_sclr = scaler.fit_transform(self.X)
        X_train = scaler.fit_transform(self.X_train)
        X_test = scaler.transform(self.X_test)
        clf = KNeighborsClassifier(n_neighbors = K).fit(X_train,self.y_train)
        yhat = clf.predict(X_test)
        acc=metrics.accuracy_score(self.y_test, yhat)
        print( "accuracy is", round(acc*100, 2), "% with k =",K)
        cm = metrics.confusion_matrix(self.y_test, yhat)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        plt.show()
        return yhat



    def plot(self,K):
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(self.X_train)
        X_test = scaler.transform(self.X_test)
        acc = []
        for n in range(1,K+1):
            clf = KNeighborsClassifier(n_neighbors = n).fit(X_train,self.y_train)
            yhat = clf.predict(X_test)
            acc.append(metrics.accuracy_score(self.y_test, yhat))

        acc = np.array(acc)
        print( "The best accuracy is", round(acc.max()*100, 2), "% with k =", acc.argmax()+1) 
        plt.plot(range(1,K+1),acc,'g',label='Accuracy')
        plt.legend()
        plt.ylabel('Accuracy ')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.show()