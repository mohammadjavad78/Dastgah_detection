from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


class Kmeans_Clustering :

    def __init__ (self, data, label):
        self.data = data
        self.label = label

    def printConfusionMatrix(self):
        for i in range (0,7):
            ypred = KMeans(i+1).fit_predict(self.data)
            cm = metrics.confusion_matrix(self.label.ravel(), ypred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        plt.show()

    def printDataMetrics(self, arr, fReduction):
        reduced_data = UMAP(random_state=0).fit_transform(self.data)
        labelencoder = LabelEncoder()
        labelencoder.fit(self.label)

        print ("{:<20} {:<20} {:<20} {:<20}".format('Name','Silhouette Score',
                                                            'chcalinski_harabasz_score',
                                                            'davies_bouldin_score'))
        print(100*'=')
        
        for i in arr:
            if (fReduction):
                ypred = KMeans(i).fit_predict(reduced_data)
                usedData = reduced_data
            else:
                ypred = KMeans(i).fit_predict(self.data)
                usedData = self.data
            print ("{:<20} {:<20} {:<20} {:<20}".format("#clusters={}".format(i), 
                                                        round(metrics.silhouette_score(usedData, ypred, metric='euclidean'), 4), 
                                                        round(metrics.calinski_harabasz_score(usedData, ypred), 4),
                                                        round(metrics.davies_bouldin_score(usedData, ypred), 4)))
            print(100*'-')
        print('\n')
        print('\n')
        print ("{:<20} {:<20} {:<30} {:<30} {:<20} {:<20} {:<20} {:<20}".format('Name',
                                                                                'adjusted_rand_score',
                                                                                'adjusted_mutual_info_score',
                                                                                'normalized_mutual_info_score', 
                                                                                'homogeneity_score', 
                                                                                'completeness_score', 
                                                                                'v_measure_score', 
                                                                                'accuracy_score'))    
        print(240*'=')
        for i in arr:
            if (fReduction):
                ypred = KMeans(i).fit_predict(reduced_data)
                usedData = reduced_data
            else:
                ypred = KMeans(i).fit_predict(self.data)
                usedData = self.data
            print("{:<20} {:<20} {:<30} {:<30} {:<20} {:<20} {:<20} {:<20}".format("#clusters={}".format(i), 
                                                        round(metrics.adjusted_rand_score(self.label, ypred), 4),
                                                        round(metrics.adjusted_mutual_info_score(self.label, ypred), 4),
                                                        round(metrics.normalized_mutual_info_score(self.label, ypred), 4),
                                                        round(metrics.homogeneity_score(self.label, ypred), 4),
                                                        round(metrics.completeness_score(self.label, ypred), 4),
                                                        round(metrics.v_measure_score(self.label,ypred), 4), 
                                                        round(metrics.accuracy_score(self.label,ypred), 4)))
            print(240*'-')
        print('\n')
    
    def plotClusters(self, nClusters, fReduction):
        reduced_data = UMAP(random_state=0).fit_transform(self.data)
        labelencoder = LabelEncoder()
        labelencoder.fit(self.label)

        lengthFigure = nClusters*5.5
        fig, axe = plt.subplots( nClusters, 2, figsize=(20,lengthFigure))
        for i in range(0,nClusters):
            ypred_reducedData = KMeans(i+1).fit_predict(reduced_data)
            y_pred_data = KMeans(i+1).fit_predict(self.data)
            if (fReduction):
                hueType = ypred_reducedData
                y_pred = ypred_reducedData
                figureName = ' cluster(s) (using UMAP feaure reduction method)'
            else:
                hueType = y_pred_data
                y_pred =y_pred_data
                figureName = ' cluster(s)'
            sns.heatmap(metrics.cluster.contingency_matrix(self.label ,y_pred ), annot=True, cbar=False, 
                                                                                ax=axe.flat[i*2], 
                                                                                yticklabels=labelencoder.classes_, fmt='d')
            sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=hueType, palette='Set3', ax=axe.flat[i*2+1])
            #ax.yaxis.set_ticklabels(rotation=0, ticklabels=labelencoder.classes_)
            axe.flat[i*2+1].set_xticks(())
            axe.flat[i*2+1].set_yticks(())
            axe.flat[i*2+1].set_title(str(i+1)+' cluster(s) (using UMAP feaure reduction method)')
            axe.flat[i*2].set_title(str(i+1)+ figureName)
        
        plt.tight_layout()
        plt.show()
    
    def plotClustersUsingArr(self, arr, fReduction):
        reduced_data = UMAP(random_state=0).fit_transform(self.data)
        labelencoder = LabelEncoder()
        labelencoder.fit(self.label)
        lengthFigure = len(arr)*5.5
        fig, axe = plt.subplots( len(arr),2, figsize=(20,lengthFigure))
        
        for i,j in zip(arr, range(len(arr))):
            ypred_reducedData = KMeans(i).fit_predict(reduced_data)
            y_pred_data = KMeans(i).fit_predict(self.data)
            if (fReduction):
                hueType = ypred_reducedData
                y_pred = ypred_reducedData
                figureName = ' cluster(s) (using UMAP feaure reduction method)'
            else:
                hueType = y_pred_data
                y_pred =y_pred_data
                figureName = ' cluster(s)'
            sns.heatmap(metrics.cluster.contingency_matrix(self.label ,y_pred ), annot=True, cbar=False, ax=axe.flat[j*2], yticklabels=labelencoder.classes_, fmt='d')
            sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=hueType, palette='Set3', ax=axe.flat[j*2+1])
            #ax.yaxis.set_ticklabels(rotation=0, ticklabels=labelencoder.classes_)
            axe.flat[j*2+1].set_xticks(())
            axe.flat[j*2+1].set_yticks(())
            axe.flat[j*2+1].set_title(str(i)+' cluster(s) (using UMAP feaure reduction method)')
            axe.flat[j*2].set_title(str(i)+ figureName)
        
        plt.tight_layout()
        plt.show()

    def elbow(self, rangeOfClusters):
        distortions = []
        K = range(1,rangeOfClusters)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.data)
            distortions.append(kmeanModel.inertia_)
        plt.figure(figsize=(12,6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()




