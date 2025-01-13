# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:12:52 2021

@author: user
"""


'''
import sys
sys.path.append('C:\\Users\\user\\.spyder-py3\\exercises\\thesis')
from logReg_02 import separate_xValues_yValues
'''

import pandas as pd # to load and manipulate data and for One-Hot Encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs

from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree

from sklearn.model_selection import cross_val_score # for cross validation
from sklearn import metrics

import threading
import time
from multiprocessing import cpu_count
from numba import jit, prange

   

def x_y_encoded(x, y):
    
    x_cols = x.columns.tolist()
    x_cols.remove('age')
    #print(type(x_cols), x_cols)
    
    x_encoded = pd.get_dummies(x, columns = x_cols)
    #print(x_encoded.head())
    y_encoded = pd.get_dummies(y)
    #print(y_encoded.head())
    return x_encoded, y_encoded


'''
It is not imported for conveniency 
'''
def separate_xValues_yValues(csvName, y_name, attrDrop, flag_to_numpy, flag_xValNames):
    df_xValues = pd.read_csv(csvName, delimiter = ',')
    
    df_xValues.drop(columns=attrDrop, inplace=True)
    df_yValue = df_xValues[y_name]
    df_xValues.drop(labels=y_name, axis=1, inplace=True)
    #print(df_yValue.name)
    #print(df_xValues.columns)
    
    if flag_to_numpy:#flag 1 or 0
        if flag_xValNames:
            return df_xValues.to_numpy(), df_yValue.to_numpy(), df_xValues.columns
        return df_xValues.to_numpy(), df_yValue.to_numpy()
    else:
        #print(flag_to_numpy)
        return df_xValues, df_yValue    

    
    
'''
Threading
'''
class alphaThread(threading.Thread):
    def __init__(self, name, sub_ccp_alphas, x_train, y_train, cross_valid, alpha_loop_values):
        threading.Thread.__init__(self)
        self.name=name
        self.sub_ccp_alphas=sub_ccp_alphas
        self.x_train=x_train
        self.y_train=y_train
        self.cross_valid=cross_valid
        self.alpha_loop_values=alpha_loop_values

    # function optimized to run on cpu 
    @jit(target ="cpu",parallel=True)    
    #@vectorize
    def run(self):
        for i in prange(len(self.sub_ccp_alphas)):
            clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='log2', random_state=0,  ccp_alpha = self.sub_ccp_alphas[i] )
            #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='sqrt', random_state=0,  ccp_alpha = self.sub_ccp_alphas[i] )
            #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None, random_state=0,  ccp_alpha = self.sub_ccp_alphas[i] )          
            #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='log2', random_state=0,  ccp_alpha = self.sub_ccp_alphas[i])
            #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='sqrt', random_state=0,  ccp_alpha = self.sub_ccp_alphas[i] )
            #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features=None, random_state=0,  ccp_alpha = self.sub_ccp_alphas[i])
            scores = cross_val_score(clf_dt, self.x_train, self.y_train, cv=self.cross_valid) #accuracies
            self.alpha_loop_values.append( [self.sub_ccp_alphas[i], np.mean(scores), np.std(scores)] )
  
    
  
@jit(target ="cpu",parallel=True)      
#@vectorize
def call_alphaThreads(ccp_alphas, x_train, y_train, cross_valid, alpha_loop_values):
    alpha_threads=[]
    num_threads=cpu_count() 
    sub_ccp_alphas = [ [] for i in range(num_threads)   ]
    
    for i in prange(len(ccp_alphas)):
        sub_ccp_alphas[ i % num_threads].append(ccp_alphas[i])
        
    for i in prange(num_threads):
        alpha_threads.append( alphaThread(i,sub_ccp_alphas[i], x_train, y_train, cross_valid, alpha_loop_values) )
        alpha_threads[i].start()
    
    for i in prange(num_threads):
        alpha_threads[i].join()
    
    return num_threads



'''
drawing Trees
'''
def draw_scores_candAlphas(alpha_loop_values):
    ## Now we can draw a graph of the means and standard deviations of the scores
    ## for each candidate value for alpha
    alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
    alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
    plt.show()
    plt.close('all')
    plt.clf() 
    
def draw_tree(clf_dt, figSize, classNames, featureNames):
    plt.figure(figsize=figSize)
    plot_tree(clf_dt, filled=True, rounded=True, class_names=classNames, feature_names=featureNames) 
    plt.show()
    plt.close('all')
    plt.clf()  

    

@jit(target ="cpu",parallel=True)     
#@vectorize
def decisionMaking_by_tree(x_train, y_train, x_test, y_test, yName, perc_valid,cut_perc,k,file):
    
    clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='log2')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='sqrt')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None)          
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='log2')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='sqrt')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features=None)
    
    treeFit_predictAccur(clf_dt, x_train, y_train, x_test, y_test, False,  yName,perc_valid,cut_perc,k,file)
    '''
    if yName!='icu':
        draw_tree(clf_dt, (15, 7.5), ["No", "Yes"], x_train.columns)
    '''
    print("finding candidates...")
           
    alpha_loop_values=candidateAlphas_scores(clf_dt, x_train, y_train, 5,file)
    #draw_scores_candAlphas(alpha_loop_values)
    print("waiting for optimal alpha, pruned tree...")         
    clf_dt_pruned = optimalAlpha_pruneTree(alpha_loop_values)
    treeFit_predictAccur(clf_dt_pruned, x_train, y_train, x_test, y_test, True,  yName, perc_valid, cut_perc, k, file)
    #draw_tree(clf_dt_pruned, (15, 7.5), ["No", "Yes"], x_train.columns)
    
    
    
@jit(target ="cpu",parallel=True)     
#@vectorize
def treeFit_predictAccur(clf_dt, x_train, y_train, x_test, y_test, flag, yName, perc_valid, cut_perc,k, file):  
    # Train Decision Tree Classifer
    clf_dt = clf_dt.fit(x_train, y_train)
    if flag:
        
        s="Y variable: "+yName+"\rTraining set: "+str(cut_perc)+"% * "+str(100-perc_valid)+"%\r"
        print(s); file.write(s)
        if k:
            s="Validation set: "+str(cut_perc)+"% /// "+str(perc_valid)+"%\r"
            print(s); file.write(s)
        else:
            s="Testing set: "+str(100-cut_perc)+"% * "+str(100-perc_valid)+"%\r"
            print(s); file.write(s) 
        
        #Predict the response for test dataset
        y_pred = clf_dt.predict(x_test)
        score = metrics.accuracy_score(y_test, y_pred)
        
        s="Accuracy: "+ str(score)+"\r"
        print(s); file.write(s)
        
        cm = metrics.confusion_matrix(y_test, y_pred)
        s="Confusion matrix: \r"+str(cm)+"\r\n\n"
        print(s); file.write(s)
    
    
    
@jit(target ="cpu",parallel=True)         
#@vectorize
def candidateAlphas_scores(clf_dt, x_train, y_train, cross_valid,file):
    
    path = clf_dt.cost_complexity_pruning_path(x_train, y_train) # determine values for alpha
    ccp_alphas = path.ccp_alphas # extract different values for alpha
    
    #print("Candidate alphas: ", ccp_alphas)
    ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha
    #print("Candidate alphas: ", ccp_alphas)
    s="Number of candidate alphas: "+str(len(ccp_alphas))+"\r"
    print(s); file.write(s)  
    
    #exclude negative alphas
    ccp_alphas_nonNegative = [alpha for alpha in ccp_alphas if alpha >= 0]
    s="Number of candidate non negative alphas: "+ str(len(ccp_alphas_nonNegative))+"\r"
    print(s); file.write(s)
    
    ## create an array to store the results of each fold during cross validiation
    alpha_loop_values = []

    start_time = time.time()
    '''
    if len(ccp_alphas!=1):
        call_alphaThreads(ccp_alphas, x_train, y_train, cross_valid, alpha_loop_values)
    else:
        scores = cross_val_score(DecisionTreeClassifier( random_state=0, ccp_alpha = ccp_alphas[0]), x_train, y_train, cv=cross_valid) #accuracies
        alpha_loop_values.append([ccp_alphas[0], np.mean(scores), np.std(scores)])
    '''
    num_threads = call_alphaThreads(ccp_alphas_nonNegative, x_train, y_train, cross_valid, alpha_loop_values)

    s="Total time spent: "+str(time.time()-start_time)+"\nNumber of threads: "+str(num_threads)+"\r"
    print(s); file.write(s)
    return alpha_loop_values



#@jit(target ="cpu") #not
def optimalAlpha_pruneTree(alpha_loop_values):
    
    alpha_loop_values.sort(key=lambda x: x[1])

    #unsorted_list.sort(key=lambda x: x[3])
    #   https://stackoverflow.com/questions/17555218/python-how-to-sort-a-list-of-lists-by-the-fourth-element-in-each-list/17555237
    
    #print(alpha_loop_values[-1])
    ideal_ccp_alpha=alpha_loop_values[-1][0]
    #print(type(ideal_ccp_alpha))
    
    return DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='log2', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='sqrt', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None, ccp_alpha=ideal_ccp_alpha)          
    #return DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='log2', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='sqrt', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='random',max_features=None, ccp_alpha=ideal_ccp_alpha)



def extract_for_test(x_train, y_train, x_test, y_test, n, m,flag_to_numpy):

    if flag_to_numpy:
        return x_train[:n, :x_train.shape[1]],y_train[:n], x_test[:m, :x_test.shape[1]],y_test[:m]
        # https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html
        # https://stackoverflow.com/questions/7670226/python-numpy-how-to-get-2d-array-column-length/7670325
    else:
        return x_train.head(n), y_train.head(n), x_test.head(m), y_test.head(m)



def main(file):
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)

    #csv_name='covid_cleared.csv'
    #csv_name='mexico_cleared_extra.csv'
    csv_name='mexico_cleared.csv'
    
    modelName=['icu','dead_with_icu','dead_without_icu']
    #yName=['dead_with_icu','dead_without_icu']
    
    attrDropTest= [ ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'dead','age_group','final_phase','total', 'group_first_phase', 'group_final_phase','group_total'],
                    ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'], 
                    ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group', 'icu', 'total', 'group_first_phase', 'group_final_phase','group_total','final_phase']
                  ]   
                  
    
    attrDropValid=[ ['entry_date', 'date_symptoms', 'date_died', 'dead','age_group','final_phase','total', 'group_first_phase', 'group_final_phase','group_total'],
                    ['entry_date', 'date_symptoms', 'date_died', 'age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'],
                    ['entry_date', 'date_symptoms', 'date_died', 'age_group', 'icu', 'total', 'group_first_phase', 'group_final_phase','group_total','final_phase']
                  ]
     
    
    if csv_name!='covid_cleared.csv':
        for i in range(len(attrDropTest)):#same with attrDropValid
            attrDropTest[i].append('Unnamed: 0.1')    
            attrDropValid[i].append('Unnamed: 0')
                
            
    perc_valid =[5,10,20]  #valid stands for validation
    #dfName=['df_cov_rest95','df_cov_rest90','df_cov_rest80']
    cut_perc=80
    for k in prange(2):
        #for i in range(len(perc_valid)):
        for perc in perc_valid:
            for j in prange(len(modelName)):
            
                if modelName[j]!='icu':
                    y_name='dead'
                else:
                    y_name=modelName[j]
                
                x_train, y_train = separate_xValues_yValues('sample_train_'+ y_name +str(perc)+'_without_index.csv', y_name, attrDropTest[j],1,False)
                if k==0:
                    x_test, y_test = separate_xValues_yValues('sample_test_'+ y_name +str(perc)+'_without_index.csv', y_name, attrDropTest[j],1,False)
                else:
                    x_test, y_test = separate_xValues_yValues('df_cov_validation'+str(perc)+'_without_index.csv', y_name, attrDropValid[j],1,False)
   
                ''' 
                x_trainHead, y_trainHead, x_testHead, y_testHead = extract_for_test(x_train, y_train, x_test, y_test, 1000, 100,1)
                decisionMaking_by_tree(x_trainHead, y_trainHead, x_testHead, y_testHead, modelName[j],perc,cut_perc,k,file)                 
                '''
                decisionMaking_by_tree(x_train, y_train, x_test, y_test, modelName[j],perc,cut_perc,k,file)      
                #'''



if __name__ == "__main__":
#   https://www.quora.com/When-I-import-my-module-in-python-it-automatically-runs-all-of-the-defined-functions-inside-of-it-How-do-I-prevent-it-from-auto-executing-my-functions-but-still-allow-me-to-call-them-in-my-main-script
# https://dataaspirant.com/decision-tree-algorithm-python-with-scikit-learn/    
    f = open("total_results.txt", "a+")
    '''
    try:
        main(f)
    except:
        pass
    '''
    main(f)
    #'''
    f.close()

'''
https://stackoverflow.com/questions/52027384/how-to-check-if-cuda-is-installed-correctly-on-anaconda
https://stackoverflow.com/questions/44210656/how-to-check-if-a-module-is-installed-in-python-and-if-not-install-it-within-t

https://www.youtube.com/watch?v=3dHJ00mAQAY&ab_channel=PragmaticAILabs
https://github.com/noahgift/cloud-data-analysis-at-scale/blob/master/GPU_Programming.ipynb

https://numba.pydata.org/numba-doc/latest/user/5minguide.html

https://numba.pydata.org/numba-doc/dev/user/parallel.html

https://tobiasraabe.github.io/blog/numba-vectorize-and-guvectorize.html
'''