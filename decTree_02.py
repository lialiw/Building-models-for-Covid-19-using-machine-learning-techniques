# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:12:52 2021

@author: user
"""
import sys
sys.path.append('C:\\Users\\user\\.spyder-py3\\exercises\\thesis')
from logReg_02 import separate_xValues_yValues

import time

import pandas as pd # to load and manipulate data and for One-Hot Encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs

from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree

from sklearn.model_selection import cross_val_score # for cross validation
from sklearn import metrics


def x_y_encoded(x, y):
    
    x_cols = x.columns.tolist()
    x_cols.remove('age')
    #print(type(x_cols), x_cols)
    
    x_encoded = pd.get_dummies(x, columns = x_cols)
    #print(x_encoded.head())
    y_encoded = pd.get_dummies(y)
    #print(y_encoded.head())
    return x_encoded, y_encoded
        


def draw_scores_candAlphas(alpha_loop_values):
    ## Now we can draw a graph of the means and standard deviations of the scores
    ## for each candidate value for alpha

    alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
    alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
    plt.show()
    plt.close('all')
    plt.clf()    
  

def draw_tree(clf_dt, figSize, classNames, featureNames):
    #plt.show()
    '''
    not working, no dummies
    '''
    #print(clf_dt)
    #print(featureNames, figSize, classNames)
    plt.figure(figsize=figSize)
    plot_tree(clf_dt, filled=True, rounded=True, class_names=classNames, feature_names=featureNames) 
    plt.show()
    plt.close('all')
    plt.clf()
    
    
def candidateAlphas_scores(clf_dt, x_train, y_train, cross_valid):
    
    path = clf_dt.cost_complexity_pruning_path(x_train, y_train) # determine values for alpha
    ccp_alphas = path.ccp_alphas # extract different values for alpha
    
    #print("Candidate alphas: ", ccp_alphas)
    ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha
    #print("Candidate alphas: ", ccp_alphas)
    print("Number of candidate alphas: ", len(ccp_alphas))
    
    #exclude negative alphas
    ccp_alphas_nonNegative = [alpha for alpha in ccp_alphas if alpha >= 0]
    print("Number of candidate non negative alphas: ", len(ccp_alphas_nonNegative))
    
    ## create an array to store the results of each fold during cross validiation
    alpha_loop_values = []

    # For each NON NEGATIVE candidate value for alpha, we will run 5-fold cross validation.
    # Then we will store the mean and s.d. of the scores (the accuracy) for each call to cross_val_score in alpha_loop_values
    
    start_time = time.time()
    for alpha in ccp_alphas_nonNegative:
        clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='log2', random_state=0, ccp_alpha= alpha)
        #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='sqrt', random_state=0, ccp_alpha= alpha)
        #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None, random_state=0, ccp_alpha= alpha)          
        #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='log2', random_state=0, ccp_alpha= alpha)
        #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='sqrt', random_state=0, ccp_alpha= alpha)
        #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features=None, random_state=0, ccp_alpha= alpha)
        scores = cross_val_score(clf_dt, x_train, y_train, cv=cross_valid) #accuracies
        alpha_loop_values.append( [alpha, np.mean(scores), np.std(scores)] )
    
    print(time.time()-start_time)
    return alpha_loop_values


def optimalAlpha_pruneTree(alpha_loop_values):
    
    alpha_loop_values.sort(key=lambda x: x[1])
    #print(type(alpha_loop_values))
    
    #unsorted_list.sort(key=lambda x: x[3])

    
    #print(alpha_loop_values[-1])
    ideal_ccp_alpha=alpha_loop_values[-1][0]
    #print(type(ideal_ccp_alpha))
    
    return DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='log2', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='sqrt', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None, ccp_alpha=ideal_ccp_alpha)          
    #return DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='log2', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='sqrt', ccp_alpha=ideal_ccp_alpha)
    #return DecisionTreeClassifier(criterion='entropy',splitter='random',max_features=None, ccp_alpha=ideal_ccp_alpha)
    
 

def treeFit_predictAccur(clf_dt, x_train, y_train, x_test, y_test, flag, yName, perc_valid, cut_perc,k):  
    # Train Decision Tree Classifer
    f = open("total_results.txt", "a+")
    

    clf_dt = clf_dt.fit(x_train, y_train)

    
    if flag:
    
        s="Y variable: "+yName+"\rTraining set: "+str(cut_perc)+"% * "+str(100-perc_valid)+"%\r"
        print(s); f.write(s)
        if k:
            s="Validation set: "+str(cut_perc)+"% /// "+str(perc_valid)+"%"+"%\r"
            print(s); f.write(s)
        else:
            s="Testing set: "+str(100-cut_perc)+"% * "+str(100-perc_valid)+"%"+"%\r"
            print(s); f.write(s) 
        
        #Predict the response for test dataset
        y_pred = clf_dt.predict(x_test)
        score = metrics.accuracy_score(y_test, y_pred)
        
        s="Accuracy: "+ str(score)+"\r"
        print(s); f.write(s)
        cm = metrics.confusion_matrix(y_test, y_pred)
        s="Confusion matrix: \r"+str(cm)+"\r\n\n"
        print(s); f.write(s)
    
    f.close()

    
        
def extract_for_test(x_train, y_train, x_test, y_test, n, m):
    return x_train.head(n), y_train.head(n), x_test.head(m), y_test.head(m)



def decisionMaking_by_tree(x_train, y_train, x_test, y_test, yName, perc_valid,cut_perc,k):
    
    clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='log2')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features='sqrt')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None)          
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='log2')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='sqrt')
    #clf_dt = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features=None)
    
    treeFit_predictAccur(clf_dt, x_train, y_train, x_test, y_test, False,  yName,perc_valid,cut_perc,k)
    #print(type(x_train))
    #draw_tree(clf_dt, (15, 7.5), ["No", "Yes"], x_train.columns)
    
    print("finding candidates...")            
    alpha_loop_values=candidateAlphas_scores(clf_dt, x_train, y_train, 5)
    #draw_scores_candAlphas(alpha_loop_values)
    
    print("waiting for optimal alpha, pruned tree...")            
    clf_dt_pruned = optimalAlpha_pruneTree(alpha_loop_values)
    treeFit_predictAccur(clf_dt_pruned, x_train, y_train, x_test, y_test, True,  yName, perc_valid, cut_perc, k)
    #draw_tree(clf_dt_pruned, (15, 7.5), ["No", "Yes"], x_train.columns)
    
    
    
def main():
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
    for k in range(2):
        #for i in range(len(perc_valid)):
        for perc in perc_valid:
            for j in range(len(modelName)):
            
                if modelName[j]!='icu':
                    y_name='dead'
                else:
                    y_name=modelName[j]
                
                x_train, y_train = separate_xValues_yValues('sample_train_'+ y_name +str(perc)+'_without_index.csv', y_name, attrDropTest[j],0,False)
                if k==0:
                    x_test, y_test = separate_xValues_yValues('sample_test_'+ y_name +str(perc)+'_without_index.csv', y_name, attrDropTest[j],0,False)
                else:
                    x_test, y_test = separate_xValues_yValues('df_cov_validation'+str(perc)+'_without_index.csv', y_name, attrDropValid[j],0,False)
   
 
                '''
                x_trainHead, y_trainHead, x_testHead, y_testHead = extract_for_test(x_train, y_train, x_test, y_test, 5000, 500)
                decisionMaking_by_tree(x_trainHead, y_trainHead, x_testHead, y_testHead, modelName[j],perc,cut_perc,k)   

                '''
                decisionMaking_by_tree(x_train, y_train, x_test, y_test, modelName[j],perc,cut_perc,k)      
                #''' 
  

if __name__ == "__main__":
#   https://www.quora.com/When-I-import-my-module-in-python-it-automatically-runs-all-of-the-defined-functions-inside-of-it-How-do-I-prevent-it-from-auto-executing-my-functions-but-still-allow-me-to-call-them-in-my-main-script
    main()