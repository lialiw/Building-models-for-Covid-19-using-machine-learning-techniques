# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:19:11 2021

@author: user
"""

import sys
sys.path.append('C:\\Users\\user\\.spyder-py3\\exercises\\thesis')

from extractData_02 import dataframeToCsv
import pandas as pd
import os

def extract_xPercent(df, xPercent, name1, name2, flag=True):
    
    df_xPerc= df.sample(frac = xPercent)
    dataframeToCsv(df_xPerc, name1)
    '''
    If flag==True
    We take the 'difference' (rest (100-xPercent%) between the initial dataframe and 
    its xPercent% (from above) and we save it to name2.csv
    '''
    if flag:
        df_100_xPerc = df.drop(df_xPerc.index)
        dataframeToCsv(df_100_xPerc, name2)



def divide_dataframe_onAttr(df, attr, value):
    df_valueOff = df.drop(df[ (df[attr]==value) ].index)
    df_valueOn = df.drop(df_valueOff.index)
    return df_valueOn, df_valueOff

    

def sampleSize(initialSize, percentCut, percentDistr):
    sizeYes = round(initialSize*percentCut*percentDistr)
    return sizeYes, round(initialSize*percentCut)-sizeYes



def createAttrSample_train_test(csv_name, attr_name, attr_value, cutPerc, attrPerc, validPerc, replace,file):
    
    df = pd.read_csv(csv_name, delimiter = ',')
    #print(len(df))

    df_Yes, df_No = divide_dataframe_onAttr(df, attr_name,  attr_value)
    #print(len(df_Yes), len(df_No))
    sampleSize_Yes, sampleSize_No = sampleSize(len(df), cutPerc, attrPerc)
    
    s = attr_name+': sampleSize_Yes = '+ str(sampleSize_Yes)+', sampleSize_No = '+ str(sampleSize_No)+'\r'
    print(s);file.write(s)
    
    
    if replace:
        sample_Yes = df_Yes.sample(n=sampleSize_Yes, replace=True)
        sample_No = df_No.sample(n=sampleSize_No,replace=True)
    else:
        sample_Yes = df_Yes.sample(n=sampleSize_Yes)
        sample_No = df_No.sample(n=sampleSize_No)



    sample_attr_train = pd.concat([sample_Yes, sample_No],sort=False).sort_index()
    '''
    #Equivalently:
    sample_attr_train = sample_No.append(sample_Yes, ignore_index=True)
    '''
    #print(type(sample_attr_train))
    #print(len(sample_attr_train), len(sample_Yes))  
    dataframeToCsv(sample_attr_train, 'sample_train_'+attr_name+str(int(validPerc*100)))
    
    sample_attr_test = df.drop(sample_attr_train.index)
    dataframeToCsv(sample_attr_test, 'sample_test_'+attr_name+str(int(validPerc*100)))
    
    s= "Cut percentage: "+ str(cutPerc)+" Validation percentage: "+ str(validPerc)+ " Attribute's yes percentage: "+ str(attrPerc)+ " Replace: "+ str(replace)+"\r"+attr_name+ ' train: '+ str(len(sample_attr_train))+' test: '+ str(len(sample_attr_test))+'\r\r'                              
    print(s);file.write(s)

    
   
def main():
    
    fName="00_sampling.txt"
    if os.path.exists(fName):
        os.remove(fName)
   
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    
    '''
    Attention !!!
    '''
    replace=True    
    covid_kaggle=False
    covid_mex_extra=True
    Yes_No_0_1=True
    
    if replace:
        if covid_kaggle:
            #df_cov : data_frame_covid, from kaggle
            #df_cov=pd.read_csv('covid_cleared.csv', delimiter = ',')
            df_cov=pd.read_csv('covid_cleared_without_index.csv', delimiter = ',')
            icuPerc=0.15 #replace=True
            deadPerc=0.3 #replace=True
        elif covid_mex_extra:
            df_cov=pd.read_csv('mexico_cleared_extra_without_index.csv', delimiter = ',')
            icuPerc=0.08 #replace=True
            deadPerc=0.43 #replace=True
        else:
            df_cov=pd.read_csv('mexico_cleared_without_index.csv', delimiter = ',')
            icuPerc=0.1 #replace=True
            deadPerc=0.47 #replace=True
    else:
        if covid_kaggle:
            #df_cov : data_frame_covid, from kaggle
            #df_cov=pd.read_csv('covid_cleared.csv', delimiter = ',')
            df_cov=pd.read_csv('covid_cleared_without_index.csv', delimiter = ',')
            icuPerc=0.066 #0.066347 #replace=False 
            deadPerc=0.252 #0.251639 #replace=False
        elif covid_mex_extra:
            df_cov=pd.read_csv('mexico_cleared_extra_without_index.csv', delimiter = ',')
            icuPerc=0.066 #0.065556 #replace=False
            deadPerc=0.309 #0.308821 #replace=False
        else:
            df_cov=pd.read_csv('mexico_cleared_without_index.csv', delimiter = ',')
            icuPerc=0.066 #0.065561 #replace=False
            deadPerc=0.308 #0.308821 #replace=False
        
    file = open(fName, "a+")
    
    s=str(len(df_cov))+'\r'
    print(s);file.write(s)
    #print(df_cov.columns)

    
    percentage_valid =[0.05,0.1,0.2]  #valid stands for validation
    dfName=['df_cov_rest95','df_cov_rest90','df_cov_rest80']
    cutPerc=0.8
    

    for i in range(len(percentage_valid)):    
        
        '''
        extract_xPercent is called to create to separate datasets from the initial dataframe df_cov. 
        The first set will be used as validation set, csv created is called df_cov_validation.csv. 
        Validation set consists of 5% of records of the initial dataset, these records are randomly chosen.
        '''
            
        extract_xPercent(df_cov, percentage_valid[i], 'df_cov_validation'+str(int(percentage_valid[i]*100)), dfName[i])
        #df_cov_validation = pd.read_csv('df_cov_validation'+str(int(percentage_valid[i]*100))+'.csv', delimiter = ',')
        #print(len(df_cov_validation))
        #'''
        
        if Yes_No_0_1:
            createAttrSample_train_test(dfName[i]+'.csv', 'icu', 0, cutPerc, icuPerc, percentage_valid[i],replace,file)
            createAttrSample_train_test(dfName[i]+'.csv', 'dead', 0, cutPerc, deadPerc, percentage_valid[i],replace,file)    
        else: # Yes_No_1_2
            createAttrSample_train_test(dfName[i]+'.csv', 'icu', 1, cutPerc, icuPerc, percentage_valid[i],replace,file)
            createAttrSample_train_test(dfName[i]+'.csv', 'dead', 1, cutPerc, deadPerc, percentage_valid[i],replace,file)  
    
    file.close()
    
    
    
if __name__ == "__main__":
    
    main()