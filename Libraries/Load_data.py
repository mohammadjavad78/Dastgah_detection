import pickle
import numpy as np
import pandas as pd

def loaddata(features=[],adr1='../Datas/X_591.pickle',adr2='../Datas/label_instrument_591.pickle',adr3='../Datas/label_dastgah_591.pickle',allinone_or_splited=True):
    with open(adr1, 'rb') as f: #loads the file from hard disk
        X = pickle.load(f)
        
    with open(adr2, 'rb') as f: #loads the file from hard disk
        label1 = pickle.load(f)
        
    with open(adr3, 'rb') as f: #loads the file from hard disk
        label2 = pickle.load(f)

    fit={"chroma_stft":[i for i in range(12)],"chroma_cqt":[i+12 for i in range(12)],"chroma_cens":[i+24 for i in range(12)],"melspectrogram":[i+36 for i in range(128)],"mfcc":[i+164 for i in range(20)],
        "rmse":[184],"spec_cent":[185],"spec_bw":[186],"spectral_contrast":[i+187 for i in range(7)],"spectral_flatness":[194],
        "rolloff":[195],"poly_features":[196,197],"tonnetz":[i+198 for i in range(6)],"zcr":[204],"fourier_tempogram":[i+205 for i in range(386)]}
    data_dict={f"feature{i}":X.T[i] for i in range(X.shape[1])}
    
    numoffeatures=[]
    if features!=[]:
        for i in features:
            for j in fit[i]:
                numoffeatures.append(j)
        data_dict={f"feature{i}":X.T[i] for i in numoffeatures}
        columns=[f"feature{i}" for i in range(len(numoffeatures))]
    else:
        columns=[f"feature{i}" for i in range(int(adr1.split('_')[-1].split('.')[0]))]
    
    data_dict["dastgah"]=label2
    data_dict["instrument"]=label1
    df= pd.DataFrame(data_dict,dtype=float)
    columns.append("dastgah")
    columns.append("instrument")
    df.columns=columns
    if(allinone_or_splited):
        return df
    return X,label1,label2