from sklearn import preprocessing
import pandas as pd
def preprocess(X_train,X_test):
    
    headers = X_train.columns
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = headers
    X_test = pd.DataFrame(scaler.transform(X_test))
    X_test.columns = headers
    return X_train,X_test