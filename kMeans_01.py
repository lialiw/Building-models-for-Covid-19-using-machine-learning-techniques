# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 00:26:35 2021

@author: user
"""


import sys
sys.path.append('C:\\Users\\user\\.spyder-py3\\exercises\\thesis')
from logReg_02 import separate_xValues_yValues


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets.samples_generator import make_blobs
from kneed import KneeLocator # https://pypi.org/project/kneed/

from numba import prange
import time

def draw_blobs():
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=0.60, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], s=50);


      
def elbow_method(start,fin,X_scaled,kmeans_kwargs,k_initial):
    sse = []
    for k in prange(start, fin):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(prange(start, fin), sse, curve="convex", direction="decreasing" )
    print('elbow: ',kl.elbow)
    return kl.elbow, kl.elbow // k_initial
    '''
    kmeans = KMeans(n_clusters=kl.elbow(), **kmeans_kwargs)
    kmeans.fit(X_scaled)
    '''
    # https://realpython.com/k-means-clustering-python/#choosing-the-appropriate-number-of-clusters
    
    

def separate_on_cluster(k_intermediate, X_scaled, predictions, y):
    #start_time = time.time()
    
    separate_clusters_list = [ [] for i in prange(k_intermediate)]
    separate_y_clusters  = [ [] for i in prange(k_intermediate)]
    
    clusters, y_clusters  = [], []

    for i in prange(len(predictions)):
        separate_clusters_list[ predictions[i] ].append(X_scaled[i])
        separate_y_clusters[ predictions[i] ].append(y[i])
    #'''
    for i in prange(len(separate_clusters_list)):
       clusters.append( np.array( separate_clusters_list[i] ) )
       y_clusters.extend(separate_y_clusters[i])
    #print('y_cluster: ',type(y_clusters),len(y_clusters))
    #print(time.time()-start_time)
    return clusters, np.array(y_clusters)



def calculate_Accuracy(predictions, y):
    correct=0
    for i in prange(predictions.shape[0]):
        #print(type(predictions[i]))
        if predictions[i]==y[i]:
            correct += 1
    accur=correct/len(predictions)
    #print(accur) 
    return accur
 
       
 
def second_layer_clustering(clusters, kmeans_kwargs, columns):
    
    X_final_scaled = np.zeros((1, columns))
    
    # https://www.kite.com/python/answers/how-to-initialize-a-numpy-array-in-python
    for i in prange(len(clusters)):
        kmeans = KMeans(n_clusters= 2, **kmeans_kwargs)
        kmeans.fit(clusters[i])
        predictions = kmeans.predict(clusters[i])
        #print(predictions.shape, type(predictions), len(predictions))
        clusters[i] = np.append(clusters[i], predictions.reshape(len(predictions),1), axis=1)
        #print(clusters[i].shape)

        # https://www.kite.com/python/answers/how-to-add-a-column-to-a-numpy-array-in-python
        X_final_scaled = np.append(X_final_scaled, clusters[i], axis=0)
    
    X_final_scaled = X_final_scaled[~np.all(X_final_scaled == 0, axis=1)]
    # https://www.geeksforgeeks.org/how-to-remove-array-rows-that-contain-only-0-using-numpy/
    #print(X_final_scaled.shape)
    return X_final_scaled
    


def extract_for_test(X_sub, y_sub, n,flag_to_numpy):
    if flag_to_numpy:
        return X_sub[:n, :X_sub.shape[1]], y_sub[:n]
        # https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html
        # https://stackoverflow.com/questions/7670226/python-numpy-how-to-get-2d-array-column-length/7670325
    else:
        return X_sub.head(n), y_sub.head(n)



def kMean_Clustering(k, kmeans_kwargs, X_scaled, y):
    kmeans = KMeans(n_clusters= k, **kmeans_kwargs)
    kmeans.fit(X_scaled)
    predictions = kmeans.predict(X_scaled)
    acc = calculate_Accuracy(predictions, y)
    print(acc);
    return acc,predictions
    
    
def best_cluster_to_csv(data, attrs, csvName):
    df = pd.DataFrame(data, columns = attrs)
    df.to_csv( csvName+'_without_index.csv', sep=',',index=False)
  
    
  
def main():
    
    start_time=time.time()
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    
    csvName = ['covid_cleared_without_index.csv','mexico_cleared_extra_without_index.csv', 'mexico_cleared_without_index.csv']    
    
    yName=['icu','dead','dead']
    modelName=['icu','dead_with_icu','dead_without_icu']
        

    for j in prange(len(csvName)):
        
        if csvName[j]=='covid_cleared_without_index.csv':
            attrDropTest= [ ['entry_date', 'date_symptoms', 'date_died', 'dead','age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'],
                            ['entry_date', 'date_symptoms', 'date_died', 'age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'], 
                            ['entry_date', 'date_symptoms', 'date_died', 'age_group', 'icu', 'total', 'group_first_phase', 'group_final_phase','group_total','final_phase']
                          ]
        else:
            attrDropTest= [ ['Unnamed: 0','entry_date', 'date_symptoms', 'date_died', 'dead','age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'],
                            ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'], 
                            ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group', 'icu', 'total', 'group_first_phase', 'group_final_phase','group_total','final_phase']
                          ] 

        for i in prange(len(yName)):
            X, y, attrs = separate_xValues_yValues(csvName[j], yName[i], attrDropTest[i], 1,True)
            #X_sub, y_sub = extract_for_test(X, y, 200, 1)

            '''
            X, y = separate_xValues_yValues('covid_cleared_without_index.csv', y_name, attrDropTest[0], 0)
            print(X.info())
            print(X.isna().sum())
            '''
            
            #''' 
            scaler = MinMaxScaler()
            #X_scaled = scaler.fit_transform(X_sub)
            X_scaled = scaler.fit_transform(X)
        
            kmeans_kwargs = { "init": 'k-means++', "n_init": 600, "max_iter": 2000}
            
            k_initial=2
            #acc_initial, predictions_initial = kMean_Clustering(k_initial, kmeans_kwargs, X_scaled, y_sub)
            acc_initial, predictions_initial = kMean_Clustering(k_initial, kmeans_kwargs, X_scaled, y)    
    
            start=1; fin=11
            better_k, k_intermediate = elbow_method(start,fin,X_scaled,kmeans_kwargs,k_initial)
            #acc_intermediate, predictions_intermediate = kMean_Clustering(k_intermediate, kmeans_kwargs, X_scaled, y_sub)
            acc_intermediate, predictions_intermediate = kMean_Clustering(k_intermediate, kmeans_kwargs, X_scaled, y)
    
            #print(type(predictions))
            #print(predictions.shape)
            #print(type(X_scaled))

 
            #clusters, y_sub_new = separate_on_cluster(k_intermediate, X_scaled, predictions_intermediate, y_sub)
            clusters, y_new = separate_on_cluster(k_intermediate, X_scaled, predictions_intermediate, y)
    
            X_final_scaled = second_layer_clustering(clusters, kmeans_kwargs, X_scaled.shape[1]+1)
            predictions_final= X_final_scaled [:, X_final_scaled.shape[1]-1 ]
            # https://stackoverflow.com/questions/8386675/extracting-specific-columns-in-numpy-array
    
            #acc_final=calculate_Accuracy( predictions_final,  y_sub_new) 
            acc_final=calculate_Accuracy( predictions_final,  y_new)     
            print(acc_final)
            print(time.time()-start_time)
        
        
            #print(type(X_final_scaled), type(predictions_initial), type(X_scaled))
            
            attrs.extend(['cluster', yName[i]])
            csv_name = '_'+ csvName[j].replace('_without_index.csv', '')

            #'''
            if acc_initial>acc_final:
                X_scaled = np.append(X_scaled, predictions_initial.reshape(len(predictions_initial),1), axis=1)
                #X_scaled = np.append(X_scaled, y_sub.reshape(len(y_sub),1), axis=1)
                X_scaled = np.append(X_scaled, y.reshape(len(y),1), axis=1)
                best_cluster_to_csv(X_scaled, attrs, 'cluster_'+ modelName[i]+ csv_name)
            else:
                #X_final_scaled = np.append(X_final_scaled, y_sub_new.reshape(len(y_sub_new),1), axis=1)
                X_final_scaled = np.append(X_final_scaled, y_new.reshape(len(y_new),1), axis=1)
                best_cluster_to_csv(X_final_scaled, attrs, 'cluster_'+ modelName[i]+ csv_name)
            #'''
            

        
if __name__ == "__main__":
    main()


# https://www.datacamp.com/community/tutorials/k-means-clustering-python?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034352&utm_targetid=aud-392016246653:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=9061579&gclid=Cj0KCQjwgtWDBhDZARIsADEKwgO9EERhikIWfEenqHXHFWiagOYvz3_wQnSAUHIjjt3c-csfeGDkEhgaAk7JEALw_wcB