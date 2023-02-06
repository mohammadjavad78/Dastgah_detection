from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

class LR:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X,self.y)
    def accurate(self,X_test,y_test):
        ypr=self.logreg.predict(X_test)
        print(metrics.classification_report(y_test,ypr))

        