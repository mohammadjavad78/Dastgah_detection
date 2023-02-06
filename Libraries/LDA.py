import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler




class lda:
    def __init__(self,X,Y,n=-1):
        self.X=X
        self.Y=Y
        self.sc=StandardScaler()
        self.X = self.sc.fit_transform(self.X)
        if(n==-1):
            n=len(self.Y.value_counts())-1
        self.lda = LDA(n_components=n)

    def LDAout(self,n=-1):
        if(n!=-1):
            self.lda = LDA(n_components=n)
        self.X = self.lda.fit_transform(self.X, self.Y)
        return self.X,self.Y

    def LDA_changeX(self,X_test):
        X_test = self.sc.transform(X_test)
        X_test = self.lda.transform(X_test)
        return X_test
