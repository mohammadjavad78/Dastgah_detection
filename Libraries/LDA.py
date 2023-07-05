import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd



class lda:
    def __init__(self,X,Y,n=-1):
        self.X=pd.DataFrame(X)
        self.Y=pd.DataFrame(Y)
        if(n==-1):
            self.n=len(self.Y.value_counts())-1
        self.lda = LDA(n_components=self.n)

    def LDAout(self):
        X = pd.DataFrame(self.lda.fit_transform(self.X, self.Y.values.ravel()),columns=[f"feature{i}" for i in range(self.n)])
        return X

    def LDA_changeX(self,X_test):
        X_test = pd.DataFrame(self.lda.transform(X_test),columns=[f"feature{i}" for i in range(self.n)])
        return X_test
