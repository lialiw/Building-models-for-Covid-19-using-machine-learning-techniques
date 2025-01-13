# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:16:36 2022

@author: Fik
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 05:06:27 2022

@author: Fik
"""

'''
    Sources:
        
        https://towardsdatascience.com/understanding-random-forest-58381e0602d2
        https://builtin.com/data-science/random-forest-algorithm
        https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
        
        https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
        https://www.datacamp.com/community/tutorials/random-forests-classifier-python    
        
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
        https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline
        
        Models:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        
        Metrics:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
                
        https://pandas.pydata.org/docs/user_guide/merging.html
        
        https://www.w3schools.com/python/python_file_remove.asp
        
        https://www.youtube.com/watch?v=Xz0x-8-cgaQ&ab_channel=StatQuestwithJoshStarmer
        https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=48&ab_channel=StatQuestwithJoshStarmer
        https://www.youtube.com/watch?v=sQ870aTKqiM&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=48&ab_channel=StatQuestwithJoshStarmer
                  
'''

import os

import pandas as pd
import numpy as np

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import matplotlib.pyplot as plt
#%matplotlib
import seaborn as sns

import copy, time


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
    
    
def extract_for_test(x_train, y_train, x_val, y_val, n, l):
    '''
    changed from: def extract_for_test(x_train, y_train, x_test, y_test, x_val, y_val, n, m, l)
    in: randomForest_01a.py
    '''    
    return x_train.head(n), y_train.head(n), x_val.head(l), y_val.head(l)
  

def feature_importance_nn_visualization(clf, attrs, importance_floor, resFileName, xlsx_path = '',  img_path= ''):

    feature_imp = pd.Series(clf.feature_importances_, index = attrs).sort_values(ascending=False)
    #print(feature_imp, type(feature_imp))
    #print(feature_imp.index)
    
    features_to_drop_list = []
    for i  in range(len(feature_imp)):
        if feature_imp[i] < importance_floor:
            features_to_drop_list.append(feature_imp.index[i])
    
    if xlsx_path!='':
        xlsxName = "feature_imp_"+resFileName+'.xlsx'
        
        feature_imp.to_excel(os.path.join(xlsx_path, xlsxName)) #https://stackoverflow.com/questions/68923773/read-and-write-xlsx-file-from-pandas-dataframe-to-specific-directory       
    
    if img_path !='':
        
        imgName = "feature_imp_"+resFileName+'.png'
        
        #Creating a bar plot
        snsBarplot = sns.barplot(x=feature_imp, y=feature_imp.index)
        sns_fig = snsBarplot.get_figure() # https://www.geeksforgeeks.org/how-to-save-seaborn-plot-to-a-file-in-python/
                
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend() 
        
        sns_fig.savefig(os.path.join(img_path, imgName), bbox_inches="tight", dpi=300, transparent=True) # https://stackoverflow.com/questions/49201174/the-seaborn-not-saving-the-whole-figure-but-only-part-of-it

        plt.show()
        
        #plt.savefig(pngName, dpi=300, format='png', transparent=True, bbox_inches='tight')
        plt.close('all')
        plt.clf()   
    
    return  features_to_drop_list  


def createParamGrid(numEstim_list, maxDepth_list, start_stop_flag, min_samples_split_list = [], min_samples_leaf_list = []):
    
    if start_stop_flag:
        # Number of trees in random forest
        n_estimators_list = [int(x) for x in np.linspace(start = numEstim_list[0], stop = numEstim_list[1], num = numEstim_list[2])]
        #print(type(n_estimators_list ))
        
        # Maximum number of levels in tree
        maxDepth_list = [int(x) for x in np.linspace(maxDepth_list[0], maxDepth_list[1], num = maxDepth_list[2])]
        #print(type(maxDepth_list))
    else:
        n_estimators_list = numEstim_list
        maxDepth_list = maxDepth_list
    
   
    if not min_samples_split_list:
        # Minimum number of samples required to split a node
        min_samples_split_list = [2, 5, 10]
    
    if not min_samples_leaf_list:
        # Minimum number of samples required at each leaf node
        min_samples_leaf_list = [1, 2, 4]
    
    # Number of features to consider at every split
    max_features_list = ['log2', 'sqrt']
    
    # Method of selecting samples for training each tree
    bootstrap_list = [True, False]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators_list, 'max_features': max_features_list,
                   'max_depth': maxDepth_list, 'min_samples_split': min_samples_split_list,
                   'min_samples_leaf': min_samples_leaf_list, 'bootstrap': bootstrap_list}

    print(param_grid , type(param_grid))
    return param_grid

    
def train_randomForest_model(x_train, y_train, param_grid,  cv_num,  scor_str,  iter_num=0, verb = 2):
    '''
    verbose: int
        >1 : the computation time for each fold and parameter candidate is displayed;
        >2 : the score is also displayed;
        >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
    
    n_jobsint, default=None
        Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
    '''
    random_state = 42
    clf = RandomForestClassifier(random_state = random_state)
    clf.fit(x_train, y_train)
    
    if iter_num:
        clf_custom = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_iter = iter_num, scoring = scor_str,  cv = cv_num, verbose = verb, n_jobs=-1, return_train_score=True)
    else:
        clf_custom = GridSearchCV(estimator = clf, param_grid = param_grid, scoring = scor_str,  cv = cv_num, verbose = verb, n_jobs=-1,  return_train_score=True)
        
    
    clf_custom.fit(x_train, y_train)
    clf_custom_best_params = clf_custom.best_params_
    #print(clf_custom_best_params, '\n')
    
    return clf, clf_custom.best_estimator_, clf_custom_best_params 

    
def val_randomForest_model(x_val, y_val, clf, img_path='', imgName='', y_val_pred_flag = False):
    
    y_val_pred = clf.predict(x_val)
    
    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_val, y_val_pred)
    prec = metrics.precision_score(y_val, y_val_pred)
    recall = metrics.recall_score(y_val, y_val_pred)
    f1 = metrics.f1_score(y_val, y_val_pred)
    #print(" Accuracy val: ", acc, " Prec val: ", prec, " Recall val: ", recall, " F1: ", f1, '\n')
    
    if img_path!='':
        plot_cm(clf, y_val, y_val_pred, img_path = img_path, imgName = imgName)
    
    if y_val_pred_flag:
        return acc, prec, recall, f1, y_val_pred
    
    return acc, prec, recall, f1
    

def plot_cm(clf, y_val, y_val_pred, img_path='', imgName=''):
    
    cm = metrics.confusion_matrix(y_val, y_val_pred, labels = clf.classes_)
    #print(cm) # https://www.w3schools.com/python/python_ml_confusion_matrix.asp
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [1,2])
    
    disp.plot( )
    if img_path!='':
        #print(img_abs_path)
        #img_path = os.path.abspath(img_path)
        plt.savefig(os.path.join(img_path, imgName)) # https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
        
    plt.show()
    plt.close('all'); plt.clf()   
    
        
def main():
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
        
        
    modelName=['icu','dead_with_icu','dead_without_icu']
    
    attrDrop_list = [ ['Unnamed: 0', 'patient_type', 'entry_date', 'date_symptoms', 'date_died', 'dead', 'age_group', 'final_phase', 'total', 'group_first_phase', 'group_final_phase','group_total'],
                      ['Unnamed: 0', 'patient_type', 'entry_date', 'date_symptoms', 'date_died', 'age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'],
                      ['Unnamed: 0', 'patient_type', 'entry_date', 'date_symptoms', 'date_died', 'age_group', 'icu', 'total', 'group_first_phase', 'group_final_phase','group_total','final_phase']
                    ]
          
    
    perc_valid =[5,10,20]  #valid stands for validation
    #dfName=['df_cov_rest95','df_cov_rest90','df_cov_rest80']

    #headFlag = True
    headFlag = False
    
    # create grid for random search
    numEstim_list = [200, 2000, 10]; maxDepth_list = [10, 100, 10]
    param_grid_randSearch = createParamGrid(numEstim_list, maxDepth_list, start_stop_flag=True)
    #print(param_grid_randSearch)
    
    # create grid for grid search
    numEstim_list = [500, 750]; maxDepth_list = [25, 50, 75]
    param_grid_searchAll = createParamGrid(numEstim_list, maxDepth_list, start_stop_flag=False)
    #print(param_grid_searchAll)
    
    cv_num = 10; iter_num = 50
    
    importance_floor = 0.03

    colNames_list = ['n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_depth', 'bootstrap', 'Accuracy', 'Precision', 'Recall', 'F1', 'base Accuracy', 'base Precision', 'base Recall', 'base F1', 'improve_acc']
    idx_list, randSearch_res_lol , searchAll_res_lol = [], [], []
    
    #'''
    img_path_rand_featImp = r"G:\UoM\publication\imgs\results\feature_imp\rand" # https://stackoverflow.com/questions/37400974/error-unicode-error-unicodeescape-codec-cant-decode-bytes-in-position-2-3
    img_path_all_featImp  = r"G:\UoM\publication\imgs\results\feature_imp\all"
    img_path_rand_cm= r"G:\UoM\publication\imgs\results\confMatrix\rand"
    img_path_all_cm = r"G:\UoM\publication\imgs\results\confMatrix\all"
   
    xlsx_path_rand_featImp = r"G:\UoM\publication\results"
    xlsx_path_all_featImp = r"G:\UoM\publication\results"
    #'''
    
   
    
    k = 0
    for j in range(len(modelName)):
        for perc in perc_valid:
            results_rand_list = []
            results_all_list = []
            
            if modelName[j]!='icu':
                y_name='dead'
            else:
                y_name=modelName[j]

            #'''
            
            x_train, y_train =  separate_xValues_yValues('df_cov_rest'+str(100-perc)+'_without_index.csv', y_name, attrDrop_list[j], 0, False)
            #for col in x_train.columns: print(col) #; print(x_train['patient_type'].unique())
            x_val, y_val= separate_xValues_yValues('df_cov_validation'+str(perc)+'_without_index.csv', y_name, attrDrop_list[j], 0, False)
                
            if headFlag:
                x_train, y_train, x_val, y_val = extract_for_test(x_train, y_train, x_val, y_val, 1000, 100)
            
            start_time_00 = time.time()
            '''
            Random Grid Search
            '''
            #'''
            # train RF models doing random grid search
            clf_base, clf_randSearch, clf_randSearch_param = train_randomForest_model(x_train, y_train, param_grid_randSearch, cv_num, iter_num=iter_num, scor_str = 'f1') #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            
            # save best model's hyperparameter values
            results_rand_list += list(clf_randSearch_param.values())
            
            # validation: 1) calculate accuracy, precision, recall f1, for base model       2) save cm
            acc_base, prec_base, recall_base, f1_base = val_randomForest_model(x_val, y_val, clf_base, img_path = img_path_rand_cm,  imgName='baseRand_'+modelName[j]+str(perc)+'.png')
            
            # calculate and save feature importance, for base model
            feature_importance_nn_visualization(clf_base, x_train.columns, importance_floor, 'baseRand_'+modelName[j]+str(perc), xlsx_path = xlsx_path_rand_featImp,  img_path = img_path_rand_featImp)
            
            # validation: 1) calculate accuracy, precision, recall f1, for random search (with best parameters) model       2) save cm 
            acc_randSearch, prec_randSearch, recall_randSearch, f1_randSearch = val_randomForest_model(x_val, y_val, clf_randSearch, img_path = img_path_rand_cm, imgName = 'rand_'+modelName[j]+str(perc)+'.png')
            
            #calculate and save feature importance, for random search (with best parameters) model 
            feature_importance_nn_visualization(clf_randSearch, x_train.columns, importance_floor, 'rand_'+modelName[j]+str(perc), xlsx_path = xlsx_path_rand_featImp,  img_path = img_path_rand_featImp)
            
            # calculate any improvement in accuracy between the above models
            improve_acc = 100 * (acc_randSearch - acc_base) / acc_base#; print(modelName[j], perc, improve_acc, '\n')
            
            # save metrics of both models and improvement
            results_rand_list += [acc_randSearch, prec_randSearch, recall_randSearch, f1_randSearch, acc_base, prec_base, recall_base, f1_base, improve_acc]
            #'''
            
            total_time = time.time()-start_time_00
            print(total_time);            

            start_time_01 = time.time()            
            '''
            Grid Search All
            '''
            '''
            # train RF models doing grid search
            clf_base, clf_searchAll, clf_searchAll_param = train_randomForest_model(x_train, y_train, param_grid_searchAll, cv_num, scor_str = 'f1') #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            
            # save best model's hyperparameter values            
            results_all_list += list(clf_searchAll_param.values())
            
            # validation: 1) calculate accuracy, precision, recall f1, for base model       2) save cm            
            acc_base, prec_base, recall_base, f1_base = val_randomForest_model(x_val, y_val, clf_base, img_path = img_path_all_cm,  imgName='baseAll_'+modelName[j]+str(perc)+'.png')
            
            # calculate and save feature importance, for base model
            feature_importance_nn_visualization(clf_base, x_train.columns, importance_floor, 'baseAll_'+modelName[j]+str(perc), xlsx_path = xlsx_path_all_featImp,  img_path = img_path_all_featImp)

            # validation: 1) calculate accuracy, precision, recall f1, for search all (with best parameters) model       2) save cm                         
            acc_searchAll, prec_searchAll, recall_searchAll, f1_searchAll = val_randomForest_model(x_val, y_val, clf_searchAll, img_path = img_path_all_cm, imgName = 'all_'+modelName[j]+str(perc)+'.png')
            
            #calculate and save feature importance, for search all (with best parameters) model 
            feature_importance_nn_visualization(clf_searchAll, x_train.columns, importance_floor, 'all_'+modelName[j]+str(perc), xlsx_path = xlsx_path_all_featImp,  img_path = img_path_all_featImp)
            
            improve_acc = 100 * (acc_searchAll- acc_base) / acc_base#; print(modelName[j], perc, improve_acc, '\n')
            
            results_all_list += [acc_searchAll, prec_searchAll, recall_searchAll, f1_searchAll, acc_base, prec_base, recall_base, f1_base, improve_acc]
            #'''
            
            randSearch_res_lol.append(results_rand_list)  
            searchAll_res_lol.append(results_all_list)              
            idx_list.append(modelName[j]+'_'+str(perc))
            k+=1; print(k)
            
            total_time = time.time()-start_time_01
            print(total_time); 
        
        total_time = time.time()-start_time_00
        print(total_time); 
            
    #print(idx_list); print(len(randSearch_res_lol))
    
    df_randSearch = pd.DataFrame(randSearch_res_lol, index = idx_list, columns = colNames_list).T
    df_randSearch.to_excel('randSearch_res.xlsx')
    
    '''
    df_searchAll= pd.DataFrame(searchAll_res_lol, index = idx_list, columns = colNames_list).T
    df_searchAll.to_excel('searchAll_res.xlsx')
    #'''
        
        
if __name__ == "__main__":
#   https://www.quora.com/When-I-import-my-module-in-python-it-automatically-runs-all-of-the-defined-functions-inside-of-it-How-do-I-prevent-it-from-auto-executing-my-functions-but-still-allow-me-to-call-them-in-my-main-script

    fileNames = ['randSearch_res.xlsx','searchAll_res.xlsx']
    for fileName in fileNames:
        if os.path.exists(fileName):
            os.remove(fileName)
    
    '''
    try:
        main()
    except:
        pass
    '''
    main()
    #'''
    