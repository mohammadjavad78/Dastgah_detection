import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class pca:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
        self.X = StandardScaler().fit_transform(self.X)

    def PCAout(self,n):
        pcas = PCA(n_components=n)
        principalComponents = pcas.fit_transform(self.X)
        principalDf = pd.DataFrame(data = principalComponents, columns = [f'feature{i}' for i in range(n)])
        # finalDf = pd.concat([principalDf, self.Y], axis = 1)
        return principalDf,self.Y
