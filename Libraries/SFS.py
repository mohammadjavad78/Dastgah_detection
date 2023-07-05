import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

class SFS:
    def __init__(self,X,Y,n=-1):
        self.X=X
        self.Y=Y
        self.headers = self.X.columns
        self.n = n
        if(n==-1):
           self.n=len(self.Y.value_counts())-1

    def SFSout(self, classifier=LinearRegression(), mode=True):
        self.classify = classifier
        self.X = pd.DataFrame(data = self.X, columns = self.headers)
        self.sfs1 = sfs(self.classify, k_features=self.n, forward=mode, verbose=1, scoring='neg_mean_squared_error')
        self.sfs1 = self.sfs1.fit(self.X, self.Y)
        self.feat_names = list(self.sfs1.k_feature_names_)
        self.sel_df = self.X[self.feat_names]
        return self.sel_df, self.Y
    

    def SFS_changeX(self,X_test):
        self.feat_names = list(self.sfs1.k_feature_names_)
        self.X_test = self.sfs1.transform(X_test)
        self.X_test = pd.DataFrame(data = self.X_test, columns = self.feat_names)
        return self.X_test

