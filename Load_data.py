import pickle
import numpy as np
import pandas as pd

def loaddata(adr1='./drive/MyDrive/X.pickle',adr2='./drive/MyDrive/label1.pickle',adr3='./drive/MyDrive/label2.pickle',allinone_or_splited=True):
    with open(adr1, 'rb') as f: #loads the file from hard disk
        X = pickle.load(f)
        
    with open(adr2, 'rb') as f: #loads the file from hard disk
        label1 = pickle.load(f)
        
    with open(adr3, 'rb') as f: #loads the file from hard disk
        label2 = pickle.load(f)
    xs={f"feature{i}":X.T[i] for i in range(X.shape[1])}
    xs["dastgah"]=label2
    xs["instrument"]=label1
    df= pd.DataFrame(xs,dtype=float)
    if(allinone_or_splited):
        return df
    return X,label1,label2