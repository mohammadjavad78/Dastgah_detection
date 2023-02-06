from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

class SVM:
    def __init__(self,X,Y,testsize):
        self.X=X
        self.Y=Y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y, test_size=testsize, random_state=0, stratify=Y)

    def accurate(self,kernel='linear',decision_function_shape='ovr',degree=3,gamma="scale",coef0=1):
        scaler = preprocessing.StandardScaler()
        X_sclr = scaler.fit_transform(self.X)
        X_train = scaler.fit_transform(self.X_train)
        X_test = scaler.transform(self.X_test)
        clf = svm.SVC(kernel=kernel,class_weight='balanced',decision_function_shape=decision_function_shape,degree=degree,gamma=gamma,coef0=coef0).fit(X_train,self.y_train)
        yhat = clf.predict(X_test)
        acc=metrics.accuracy_score(self.y_test, yhat)
        print( "accuracy is", round(acc*100, 2))
        cm = metrics.confusion_matrix(self.y_test, yhat)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        plt.show()
