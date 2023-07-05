from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from CrossValidation import *

class LR:
    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test


    def accurate(self,X_test=[],Y_test=[],LDA_in=False):
        if(type(X_test)==type([])):
            X_test=self.X_test
            Y_test=self.Y_test
        self.clf = LogisticRegression(max_iter=1000)
        scores = cross(self.clf,5,self.X_train,self.Y_train,LDA_in)
        print("Cross-Validation Results: ",scores)
        print( "min validation accuracy is", round(np.min(scores)*100, 2))
        print( "mean validation accuracy is", round(np.mean(scores)*100, 2))
        print( "max validation accuracy is", round(np.max(scores)*100, 2))

        self.clf = LogisticRegression().fit(self.X_train,self.Y_train)
        self.yhat = self.clf.predict(X_test)
        self.acc=metrics.accuracy_score(Y_test, self.yhat)
        print( "====> Test accuracy is", round(self.acc*100, 2))
        cm = metrics.confusion_matrix(Y_test, self.yhat)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        plt.show()
        print(metrics.classification_report(Y_test,self.yhat))

        
        
