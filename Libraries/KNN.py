from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from CrossValidation import *

class KNN:
    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.clf = KNeighborsClassifier(n_neighbors = 1)


    def plot(self,K, LDA_in=False):
        acc = []
        score2 = []
        for n in range(1,K+1):        
            scores = cross(KNeighborsClassifier(n_neighbors = K),5,self.X_train,self.Y_train,LDA_in)
            acc.append(np.mean(scores))
            score2.append(scores)

        acc = np.array(acc)
        print( "The best accuracy is", round(acc.max()*100, 2), "% with k =", acc.argmax()+1) 
        print("Cross-Validation Results: ",score2[acc.argmax()])
        print( "min validation accuracy is", round(np.min(score2[acc.argmax()])*100, 2), "% with k =",acc.argmax()+1)
        print( "mean validation accuracy is", round(np.mean(score2[acc.argmax()])*100, 2), "% with k =",acc.argmax()+1)
        print( "max validation accuracy is", round(np.max(score2[acc.argmax()])*100, 2), "% with k =",acc.argmax()+1)
        plt.plot(range(1,K+1),acc,'g',label='Accuracy')
        plt.legend()
        plt.ylabel('Accuracy ')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.show()

        self.clf = KNeighborsClassifier(n_neighbors = acc.argmax()+1).fit(self.X_train,self.Y_train.values.ravel())
        self.yhat = self.clf.predict(self.X_test)
        self.acc=metrics.accuracy_score(self.Y_test.values.ravel(), self.yhat)
        print( "====> Test accuracy is", round(self.acc*100, 2), "% with k =",acc.argmax()+1)
        cm = metrics.confusion_matrix(self.Y_test.values.ravel(), self.yhat)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        plt.show()
        return acc.argmax()+1

    def predict(self,X_test=[]):
        if(X_test==[]):
            X_test=self.X_test
        return self.clf.predict(X_test)
