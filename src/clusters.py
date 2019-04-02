
import numpy as np
import pandas as pd
import random
np.random.seed(seed=14)

class KMeans:
    # This should return indices of customers in df who need a certain model in the ensemble applied to them
    def __init__(self,df,cols,k=6):
        self.k = k
        self.columns = cols
        self.df = df
        self.X = self.df[self.columns].values
        self.y = None
        self.centers = self.get_centers()
        self.convergence_dist = None
        self.clusters = []
        self.targets = []
        self.ensemble_Xs = []
        self.ensemble_ys = []
    
    def assign_test_clusters(self,df_test):
        
        test_clusters = []
        for x in df_test[self.columns].values:
            distances = []
            for center in self.centers:
                distances.append( np.linalg.norm(x-center) )
            test_clusters.append( np.array(distances).argmin() )
        
        for cluster in range(self.k):
            indices = np.argwhere(np.array(test_clusters)==cluster)
            ensemble_X = df_test[self.columns].iloc[indices]
            ensemble_y = df_test["Target"].iloc[indices]
            self.ensemble_Xs.append(ensemble_X)
            self.ensemble_ys.append(ensemble_y)
        

    def get_clusters(self):
        for i in range(self.k):
            mask = np.argwhere(self.y==i).ravel()

            members = self.X[mask,:]
            members_targets = self.df["Target"].values[mask]

            cluster = pd.DataFrame(data=members,columns=self.columns,index=mask)
            cluster_target = pd.DataFrame(data=members_targets,columns=["Target"],index=mask)

            self.clusters.append(cluster)
            self.targets.append(cluster_target)

    def get_centers(self):
        return random.sample(list(self.X),self.k)
           
    def classify(self):
        self.y = []

        for xi in self.X:
            k_distances = []
            
            for center in self.centers:
                k_distances.append( (np.linalg.norm(xi - center))**2 )
                
            self.y.append(np.argmin(k_distances))
        
    def calc_convergence_dist(self,center,new_center):
        return np.linalg.norm(new_center-center)
        
    def converged(self,min_converge,n_iter,max_iter):
        return ( (n_iter == max_iter) | (self.convergence_dist < min_converge) )
        
    def update_centers(self):
        self.y = np.array(self.y)
        self.X = np.array(self.X)
        new_centers = []
#         print(self.centers)
        
        for i in range(len(self.centers)):
            cluster_members = self.X[ np.argwhere(self.y==i).ravel() ]
#             print(cluster_members)
            
            new_center = []
            converge_dist = 0
            for j in range(len(cluster_members[0])):
                new_center.append( cluster_members[:,j].mean() )
            
            converge_dist += self.calc_convergence_dist(self.centers[i],new_center)  
            new_centers.append( np.array(new_center) )
                
        self.convergence_dist = converge_dist
        self.centers = new_centers
        # print(self.centers)
        
    def fit(self,min_converge=1,max_iter=5):
        self.convergence_dist = min_converge
        n_iter = 0
        
        while not self.converged(min_converge, n_iter, max_iter):
            # Calculate distance of each point, classify nearest center index in self.y
            self.classify()
            # Update centers
            self.update_centers()
            n_iter += 1
        
        self.get_clusters()
            
    def sse(self):
        sse = 0
        for i in range(len(self.X)):
            center = self.y[i]
            sse += self.calc_convergence_dist(self.X[i], self.centers[center])
        return sse
    
    # def get_k_dataframes(self):
    #     X_ks = []
    #     for i in range(self.k):
    #         mask = np.argwhere(self.y==i)
    #         X_k = self.X[mask,:]
    #         X_ks.append(X_k)
    #     return X_ks