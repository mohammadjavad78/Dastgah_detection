import pandas as pd
from sklearn.decomposition import PCA

class pca:
    def __init__(self,X):
        self.X=X

    def PCAout(self,n):
        self.pcas = PCA(n_components=n)
        self.n=n
        principalComponents = self.pcas.fit_transform(self.X)
        principalDf = pd.DataFrame(data = principalComponents, columns = [f'feature{i}' for i in range(n)])
        return principalDf

    def PCA_changeX(self,X_test):
        X_test = pd.DataFrame(self.pcas.transform(X_test), columns = [f'feature{i}' for i in range(self.n)])
        return X_test