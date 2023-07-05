from sklearn import metrics
from sklearn.model_selection import train_test_split
from LDA import *

def cross(clf,n,X,Y,lda_in):
    clfs=[]
    accs=[]
    for i in range(n):
        X_train,X_val,y_train,y_val = train_test_split(X,Y,test_size=0.2)
        clfs.append(clf)
        if(lda_in):
            Lda=lda(X_train,y_train)
            X_train=Lda.LDAout()
            X_val=Lda.LDA_changeX(X_val)
        clfs[i].fit(X_train,y_train)
        YY=clfs[i].predict(X_val)
        accs.append(metrics.accuracy_score(YY,y_val))
        # print(accs)
        
    # XX=[]
    # YY=[]
    # YP=[]
    # accs=[]
    
    # for i in range(n):
    #     XX.append(X.iloc[X.shape[0]//5*i:X.shape[0]//5*(i+1),:])
    #     YY.append(Y.iloc[Y.shape[0]//5*i:Y.shape[0]//5*(i+1),:])
    #     clfs.append(clf)
    # for i in range(n):
    #     for j in range(n):
    #         if i!=j:

    #             clfs[i].fit(XX[i],YY[i].values.ravel())
    #     YP.append(clfs[i].predict_proba(XX[i]))
    return accs