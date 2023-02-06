from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

class MLP:
    def __init__(self,X,y,testsize):
        self.X=X
        self.y=y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=testsize, random_state=0, stratify=y)
        self.clf = MLPClassifier()
        self.clf.fit(self.X_train,self.y_train)
    def accurate(self):
        ypr=self.clf.predict(self.X_test)
        print(metrics.classification_report(self.y_test,ypr))

        