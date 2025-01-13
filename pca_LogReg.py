# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:09:03 2021

@author: user

"""  
    
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Logit

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_regression


def separate_xValues_yValues(csvName, y_name, attrDrop, flag_to_numpy, flag_xValNames):
    df_xValues = pd.read_csv(csvName, delimiter = ',')
    
    df_xValues.drop(columns=attrDrop, inplace=True)
    df_yValue = df_xValues[y_name]
    df_xValues.drop(labels=y_name, axis=1, inplace=True)
    #print(df_yValue.name)
    #print(df_xValues.columns)
    
    if flag_to_numpy:#flag 1 or 0
        if flag_xValNames:
            return df_xValues.to_numpy(), df_yValue.to_numpy(), list(df_xValues.columns)
        return df_xValues.to_numpy(), df_yValue.to_numpy()
    else:
        #print(flag_to_numpy)
        return df_xValues, df_yValue   
    
    


def pValuesSummary(X,y,file,k):
    #Find p-value (significance) of coefficients
    logit_model = sm.Logit(y,X)
    result=logit_model.fit()
    if not k:        
        file.write(str(result.summary2())+"\r")
        #print(result.summary2() )
        
        #result.params ->Series
        params=result.params.index
        #print(params)
        p_val=result.summary2().tables[1]['P>|z|']
        
        params_to_be_ignored = set() #set
        for i in range(len(p_val)):
            if p_val[i]>0.05:
                params_to_be_ignored.add(params[i])
        #print(params_to_be_ignored)
        return params_to_be_ignored



def scoresCrossVal(log_reg,X,y,num_cv):
    scores = cross_val_score(log_reg , X, y, cv=num_cv)
    print('Cross-Validation Accuracy Scores', scores)
    scores = pd.Series(scores)
    print("Min:", scores.min(), " Mean:",scores.mean(), " Max:",scores.max())    
    
    
    
def evaluation(log_reg,X,y,file):
    '''
    Other ways of measuring model performance: precision, recall, F1 Score, ROC Curve, etc.
    accuracy = correct predictions / total number of data points
    '''
    # Accuracy
    score = log_reg.score(X, y)
    s="Accuracy: "+ str(score)+"\r"
    print(s); file.write(s)
        
    #Make predictions on entire test data
    predictions = log_reg.predict(X)
    # Confusion Matrix (Digits Dataset)
    cm = metrics.confusion_matrix(y, predictions)
    #confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
    s="Confusion matrix: \r"+str(cm)+"\r\n\n"
    print(s); file.write(s)
    
    
    
def logReg(solv, x_train, y_train, x_test, y_test, yName, perc_valid, cut_perc,k,file,params_to_be_dropped,roundTwo):
       
    # Make an instance of the Model
    logisticRegr = LogisticRegression(solver=solv)

    # Training the model on the data, storing the information learned from the data
    logisticRegr.fit(x_train, y_train)
    
    #scoresCrossVal(logisticRegr,x_train, y_train, 10)
    
    s="Y variable: "+yName+"\rTraining set: "+str(cut_perc)+"% * "+str(100-perc_valid)+"%\r"
    print(s); file.write(s)
    if k:
        s="Validation set: "+str(cut_perc)+"% /// "+str(perc_valid)+"%\r"
        print(s); file.write(s)
    else:
        s="Testing set: "+str(100-cut_perc)+"% * "+str(100-perc_valid)+"%\r"
        print(s); file.write(s)
        if roundTwo:          
            params_to_be_ignored = pValuesSummary(x_train,y_train,file,k)
            params_to_be_dropped[yName].append(params_to_be_ignored)
    
    #print('train',x_train.columns); print('test',x_test.columns)
    evaluation(logisticRegr, x_test, y_test, file)
    
    
    
def applyingTrainTestValid(perc_valid, modelNames, attrTest, attrValid, cut_perc, file, params_to_be_dropped, roundTwo):
    flag_to_numpy, flag_xValNames = 0, False
    
    for k in range(2):
        #for i in range(len(perc_valid)):
        for perc in perc_valid:
            for model in modelNames:
                
                if model!='icu':
                    y_name='dead'
                else:
                    y_name=model
                
                csv_name = 'sample_train_'+ y_name +str(perc)+'_without_index.csv'
                x_train, y_train = separate_xValues_yValues(csv_name, y_name, attrTest[model], flag_to_numpy, flag_xValNames)
                #print(x_train.columns)
                solv='liblinear'
                if k==0:
                    csv_name = 'sample_test_'+ y_name +str(perc)+'_without_index.csv'
                    x_test, y_test = separate_xValues_yValues(csv_name, y_name, attrTest[model], flag_to_numpy, flag_xValNames)
                else:
                    csv_name = 'df_cov_validation'+str(perc)+'.csv'#'_without_index.csv'
                    x_test, y_test = separate_xValues_yValues(csv_name, y_name, attrValid[model], flag_to_numpy, flag_xValNames)
                #print(x_test.columns)
                
                logReg(solv, x_train, y_train, x_test, y_test, model, perc, cut_perc, k, file, params_to_be_dropped,roundTwo)
                
                

def new_attrDrop_testValid(modelNames,params_to_be_dropped,attrDropTest,attrDropValid):
    new_attrDropTest=attrDropTest
    new_attrDropValid=attrDropValid
    
    for model in modelNames:
        intersection0_1 = params_to_be_dropped[model][0].intersection(params_to_be_dropped[model][1])
        intersection1_2 = params_to_be_dropped[model][1].intersection(params_to_be_dropped[model][2])
        intersection0_2 = params_to_be_dropped[model][0].intersection(params_to_be_dropped[model][2])        
        union_inter_01_12_02 =intersection0_1.union(intersection1_2).union(intersection0_2)
        #print(list(union_inter_01_12_02))
        new_attrDropTest[model].extend(list(union_inter_01_12_02))
        new_attrDropValid[model].extend(list(union_inter_01_12_02))
    return new_attrDropTest, new_attrDropValid
    


def if_pca(flag,csv_name):
    
    if flag:
        df_cov = pd.read_csv(csv_name); x_cols = list(df_cov.columns)
        df_pca = pd.read_csv('pca_Avalysis.csv')
        #print(type(df_pca.iloc[0][0]))
        attrTest, attrValid,params_to_be_dropped,modelNames = {},{},{},[]
        
        for i in range(len(df_pca)):
            dataset_name=csv_name.replace('.csv','')
            if (dataset_name +'_icu' == df_pca.iloc[i][0] ) or (dataset_name+'_dead' == df_pca.iloc[i][0]):
                key = df_pca.iloc[i][0].replace(dataset_name+'_',''); #print(key)
                #value=df_pca.iloc[i][1].strip('}{').split(', ')
                value_temp = [item.strip('\'') for item in df_pca.iloc[i][1].strip('}{').split(', ') ]; #print(value)
                # https://stackoverflow.com/questions/40950791/remove-quotes-from-string-in-python/40950987                
                
                value_temp.append(key)#essential for separate_xValues_yValues
                value = list(set(x_cols)-set(value_temp))
                attrTest.update({key: value})                
                #value.remove('Unnamed: 0')#essential for validation
                
                attrValid.update({key: value})
                
                params_to_be_dropped.update({key: []} )
                modelNames.append(key)
        #print('attrTest ',attrTest,'\nattrVal ',attrValid)
        #print(attrValid)
    
    else:
        attrTest = { 'icu': ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'dead','age_group','final_phase','total', 'group_first_phase', 'group_final_phase','group_total'],
                     'dead_with_icu': ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'], 
                     'dead_without_icu': ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group', 'icu', 'total', 'group_first_phase', 'group_final_phase','group_total','final_phase']
                   }
                
        attrValid = { 'icu': ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'dead','age_group','final_phase','total', 'group_first_phase', 'group_final_phase','group_total'],
                      'dead_with_icu': ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group','total', 'group_first_phase', 'group_final_phase','group_total','final_phase'],
                      'dead_without_icu': ['Unnamed: 0', 'entry_date', 'date_symptoms', 'date_died', 'age_group', 'icu', 'total', 'group_first_phase', 'group_final_phase','group_total','final_phase']
                    }
        if csv_name!='covid_cleared.csv':
            for key in attrTest.keys(): #same with attrValid.keys()
                attrTest[key].append('Unnamed: 0.1')
                attrValid[key].append('Unnamed: 0.1')
        
        params_to_be_dropped = {'icu':[],'dead_with_icu':[],'dead_without_icu':[]}
        modelNames=['icu','dead_with_icu','dead_without_icu']

    return attrTest, attrValid, params_to_be_dropped, modelNames 
        
        
        
def main(f_initial,f_final):
    
    roundTwo=False
    flag_to_pca=False#flag for PCA
    #csv_name='covid_cleared.csv'
    #csv_name='mexico_cleared_extra.csv'
    csv_name='mexico_cleared.csv'
    
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)

    #Initial
    attrTest, attrValid, params_to_be_dropped, modelNames = if_pca(flag_to_pca,csv_name)
    #'''
    
    perc_valid =[5,10,20]  #valid stands for validation
    #dfName=['df_cov_rest95','df_cov_rest90','df_cov_rest80']
    cut_perc=80
    
    #'''
    applyingTrainTestValid(perc_valid, modelNames, attrTest, attrValid, cut_perc, f_initial, params_to_be_dropped, roundTwo)    
    
    if roundTwo:
        #print(params_to_be_dropped)
        new_attrDropTest, new_attrDropValid = new_attrDrop_testValid(modelNames, params_to_be_dropped, attrTest, attrValid )
        applyingTrainTestValid(perc_valid, modelNames, new_attrDropTest, new_attrDropValid, cut_perc, f_final, params_to_be_dropped, roundTwo)
    #'''



if __name__ == "__main__":     
    
    file_names=["total_results_initial.txt","total_results_final.txt"]
    files = []
    for j in range(len(file_names)):
        if os.path.exists(file_names[j]):
            os.remove(file_names[j])
        files.append(open(file_names[j], "a+"))

    #'''
    main(files[0], files[1])
    '''
    try:
        main(f_initial,f_final)
    except:
        pass
    #'''
    for file in files:
       file.close()
    
