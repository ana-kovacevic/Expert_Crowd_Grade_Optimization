# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:11:41 2023

@author: akovacevic
"""

# import KMeans
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import entropy
        


def create_cluster_experts (data, model):
    # create kmeans object
    #kmeans = KMeans(n_clusters= num_experts )
    # fit kmeans object to data
    #model.fit(data)
    # print location of clusters learned by kmeans object
    clusterd_experts = model.cluster_centers_

    # save new clusters for chart
    y_km = model.fit_predict(data)
    
    return clusterd_experts, y_km

def optimize_k_number(x, kmin = 2, kmax = 10):

    best_params = {}
    best_params['n_k'] = 0
    best_params['silhouette_score'] = 0
    best_params['model'] = None
    
    silhouette_dict={}
    
    k_range=list(range(kmin, kmax))
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in k_range:
        kmeans = KMeans(n_clusters = k, init='k-means++', random_state = 0).fit(x)
        labels = kmeans.predict(x) #kmeans.labels_
        
        sil = silhouette_score(x, labels, metric = 'euclidean')
        silhouette_dict.update({k:sil})
        
        if sil > best_params['silhouette_score']:
                best_params['n_k'] = k
                best_params['silhouette_score'] = sil
                best_params['model'] = kmeans
                print ('New optimal hyperparameters for Kmeans')
        
    return best_params, silhouette_dict    

#data = voters_lookup
#clust_labels = clust_labels_all
#centroids = all_centroids
#factor_data =user_factors
'''
factor_data = df_expert
data = voters_lookup
clust_labels = clust_labels
centroids =  expert_centroids
'''
def determnin_cluster_quality(factor_data, data, clust_labels, centroids):

    
    final_data = data.copy()
    final_data[['name','group']] = final_data['voter'].str.split('_', expand = True)[[0,2]]

    final_data['clust_label'] = clust_labels
  
    
    unique_clust_labl = np.unique(clust_labels)
    
    cluster_quality = pd.DataFrame(columns = ['cluster', 'entropy', 'N', 'compactenss'])
    cluster_quality['cluster'] = unique_clust_labl
    #entropy_dict = {}
    #cl = 0
    for cl in unique_clust_labl:
       clust_data = final_data[final_data['clust_label'] == cl]
       dist = clust_data['group'].value_counts()/len(clust_data)
       probs = dist.values
       entrp = entropy(probs, base =  2)
       #entropy_dict.update({cl : entrp})
       N = len(clust_data)
       center = centroids[[cl]]
                                      
       compact  = euclidean_distances(factor_data[factor_data.index.isin(list(clust_data.index))], center)
       compact  = compact.sum()
       
       #cluster_quality["entropy"] = cluster_quality["cluster"].apply(lambda x: entropy_dict.get(x))
       
       cluster_quality.loc[cluster_quality['cluster'] == cl, 'entropy'] = entrp
       cluster_quality.loc[cluster_quality['cluster'] == cl, 'N'] = N
       cluster_quality.loc[cluster_quality['cluster'] == cl, 'compactenss'] = compact
    
    cluster_quality['avg_compact'] = cluster_quality['compactenss'] / cluster_quality['N']
    #probs = np.bincount(clust_labels)/len(clust_labels)
    return cluster_quality


def plot_clusters_3d(user_factors, expert_ids, y_km):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x0 = user_factors[expert_ids][y_km == 0,0]
    y0 = user_factors[expert_ids][y_km == 0,1]
    z0 = user_factors[expert_ids][y_km == 0,2]


    x1 = user_factors[expert_ids][y_km == 1,0]
    y1 = user_factors[expert_ids][y_km == 1,1]
    z1 = user_factors[expert_ids][y_km == 1,2]

    x2 = user_factors[expert_ids][y_km == 2,0]
    y2 = user_factors[expert_ids][y_km == 2,1]
    z2 = user_factors[expert_ids][y_km == 2,2]


    #x3 = user_factors[expert_ids][y_km == 3,0]
    #y3 = user_factors[expert_ids][y_km == 3,1]
    #z3 = user_factors[expert_ids][y_km == 3,2]

    ax.scatter(x0, y0, z0, c='r', marker='o')
    ax.scatter(x1, y1, z1, c='b', marker='^')
    ax.scatter(x2, y2, z2, c='y', marker='v')
    #ax.scatter(x3, y3, z3, c='g', marker='<')

    ax.set_xlabel('First Factor')
    ax.set_ylabel('Second Factor')
    ax.set_zlabel('Third Factor')

    plt.show()





