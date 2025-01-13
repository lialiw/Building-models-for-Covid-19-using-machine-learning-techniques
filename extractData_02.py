# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:17:41 2021

@author: user
"""

import numpy as np
import pandas as pd

def rmCols_nonCovid_translate_csv():
    
    #df_cov : dataframe of datos_abiertos_covid19.csv
    df_cov=pd.read_csv('datos_abiertos_covid19.csv', delimiter = ',')

    #print(df_cov.shape)
    #print(df_cov.columns)
    
    '''
    'FECHA_ACTUALIZACION'
    '''
    col_not_to_be_used = ['FECHA_ACTUALIZACION','ID_REGISTRO','ORIGEN','SECTOR','ENTIDAD_UM','ENTIDAD_NAC','ENTIDAD_RES','MUNICIPIO_RES','NACIONALIDAD','HABLA_LENGUA_INDIG',
                          'INDIGENA','TOMA_MUESTRA_ANTIGENO','RESULTADO_ANTIGENO','MIGRANTE','PAIS_NACIONALIDAD','PAIS_ORIGEN']
    
    #I permanently remove/drop columns whose names are included in col_not_to_be_used from df_cov
    for col_name in col_not_to_be_used:
        df_cov.drop(labels=col_name, axis=1, inplace=True)
    #print(df_cov.columns)
    
    df_cov.rename(columns = { 'SEXO':'sex', 'TIPO_PACIENTE':'patient_type', 'FECHA_INGRESO':'entry_date','FECHA_SINTOMAS':'date_symptoms', 'FECHA_DEF':'date_died', 
                             'INTUBADO':'intubed','NEUMONIA':'pneumonia','EDAD':'age', 'EMBARAZO':'pregnancy', 'DIABETES':'diabetes', 'EPOC':'copd', 'ASMA':'asthma', 
                             'INMUSUPR':'inmsupr','HIPERTENSION':'hypertension', 'OTRA_COM':'other_disease', 'CARDIOVASCULAR':'cardiovascular', 'OBESIDAD':'obesity',
                             'RENAL_CRONICA':'renal_chronic', 'TABAQUISMO':'tobacco', 'OTRO_CASO':'contact_other_covid','TOMA_MUESTRA_LAB':'covid_test', 'RESULTADO_LAB':'covid_res', 
                             'CLASIFICACION_FINAL':'class_final', 'UCI':'icu' }, inplace = True)
    
    #print(df_cov.shape)
    #print(df_cov.columns)
    dataframeToCsv(df_cov,'covid_Mexico')
    
    
    
def drop_icu_covidRes_Nan(df_cov, dataset, extra):
        
    #Filtering all rows for which 'icu' is unspecified, meamimg icu equals to 97 or 98 or 99
    df_cov.drop(df_cov[ (df_cov['icu']==97) | (df_cov['icu']==98) | (df_cov['icu']==99) ].index, inplace=True)
    #print(df_cov.shape)
    if dataset=='kaggle':
        #Filtering all rows for which 'covid_res' were (still) being awaited, meanimg icu equals to 3
        df_cov.drop(df_cov[df_cov['covid_res']==3].index, inplace=True)
        #print(df_cov.shape)
    else:
        df_cov.drop(df_cov[ (df_cov['patient_type']==99) ].index, inplace=True)
        #We cannot have records of patients who were not tested, yet they had a covid result for their 'lab test' or classified with covid.
        df_cov.drop(df_cov[ (df_cov['covid_test']!=1) ].index, inplace=True)
        
        if extra: #-->518705
            df_cov.loc[(df_cov.covid_res != 1) & (df_cov.class_final == 3), 'covid_res']=1   #anapodi periptwsi
            df_cov.loc[(df_cov.covid_res != 2) & (df_cov.class_final == 7), 'covid_res']=2
        
        df_cov.drop(df_cov[~( ((df_cov['covid_res']==1) & (df_cov['class_final']==3)) | ((df_cov['covid_res']==2) & (df_cov['class_final']==7)) )].index, inplace=True) #-->516298
        #df_cov.drop(df_cov[ ~((df_cov['covid_res']==1) | (df_cov['covid_res']==2))].index, inplace=True) #-->516299
        df_cov.drop(labels=["covid_test","class_final"], axis=1, inplace=True)
    


def createAgeAttr(df_cov):
    #Creating a list with  an age-group-id for each patient (based on age groups of EODY)     
    ageGroup=[]
    for item in df_cov['age'].iteritems(): 
        #print(item[1])
        # https://www.w3schools.com/python/python_tuples_access.asp
        if item[1] <=17:
            ageGroup.append(1)
        elif item[1] <=39:
            ageGroup.append(2)
        elif item[1] <=64:
            ageGroup.append(3)
        else:
            ageGroup.append(4)

    #Inserting this List as a column to the dataframe
    df_cov.insert(8, "age_group", ageGroup, True) 
   


def createPhasesAttr(df_cov, dataset):
    #Replacing all '9999-99-99' in 'date died with Nan
    df_cov.loc[ (df_cov['date_died'] == '9999-99-99'), 'date_died'] = np.nan

    #Converting strings of dates to 'datetime' in Pandas
    date_cols=['entry_date','date_symptoms','date_died']

    if dataset=='kaggle':
        frm='%d-%m-%Y'
    else:
        frm='%Y-%m-%d'
        
    for col in date_cols:
        df_cov[col] = pd.to_datetime(df_cov[col], format=frm)
        #print(df_cov[col])
    
    '''
    Creating for each patient the following attributes:
        'first_phase' : 'entry_date' - 'date_symptoms'
        'final_phase' : 'date_died' - 'entry_date'
        'total' : 'date_died' - 'date_symptoms'

    If 'final_phase'== Nan or 'total' == Nan, the patient didn't die
    '''
    df_cov['first_phase'] = df_cov['entry_date'] - df_cov['date_symptoms']
    df_cov['final_phase'] = df_cov['date_died'] - df_cov['entry_date']
    df_cov['total'] = df_cov['date_died'] - df_cov['date_symptoms']
   


def clearPhaseAttr(df_cov,date_cols):
    
    #Converting 'datetime' in Pandas to float (and/or Nan) values
    #Clearing data from negative differences/periods of time. Records with date of (first) symptoms or enrty date later than date of death are not considered valid.
    #Replacing Nan values with -1.0 in 'first_phase', 'final_phase', 'total', 'date_died' 
    
    for col in date_cols:
        df_cov[col]=df_cov[col].dt.days
        df_cov.drop(df_cov[ (df_cov[col]< 0.0)].index, inplace=True)
        df_cov[col] = df_cov[col].fillna(-1.0)
  
    

def createGroupPhaseAttr(df_cov,date_cols): 
    '''
    Creating for each patient the following attributes:
        'group_first_phase', 'group_final_phase','group_total' 

    'group_first_phase'/possiible values: 0 = 0 days, 1 < 10 days, 2 < 20 days, 3 < 30 days, 4 < 40 days, 5 < 50 days, 7 < 70 days, 8 >= 70 days
    'group_final_phase'/possiible values: -1 = did not die, 0 = 0 days, 1 < 10 days, 2 < 20 days, 3 < 30 days, 4 < 40 days, 5 < 50 days, 7 < 70 days, 8 >= 70 days
    'group_total'/possiible values: -1 = did not die, 0 = 0 days, 1 < 10 days, 2 < 20 days, 3 < 30 days, 4 < 40 days, 5 < 50 days, 7 < 70 days, 8 >= 70 days
    '''

    for i in range(len(date_cols)):
        aList=[]
        col=df_cov[date_cols[i]]
        for item in col.iteritems():
            if item[1]==-1:
                aList.append(-1)
            elif item[1]==0:
                aList.append(0)
            elif item[1]<10:
                aList.append(1)
            elif item[1]<20:
                aList.append(2)
            elif item[1]<30:
                aList.append(3)
            elif item[1]<40:
                aList.append(4)
            elif item[1]<50:
                aList.append(5)
            elif item[1]<70:
                aList.append(7)
            else:
                aList.append(8)
        #print(i)
        df_cov.insert(26+i, "group_"+date_cols[i], aList, True) 



def createDeadAttr(df_cov):
    #Creating a list with an indicator for each patient to determine if he died (Yes=1) or not (No=2)
    died=[]
    for item in df_cov['total'].iteritems(): 
        #print(type(item[1]))
        #nat = pd._lib.tslib.NaTType()
        if item[1]==-1.0:
            died.append(2)
        else:
            died.append(1)
            #print(died)        
            #Inserting this List as a column to the dataframe
    df_cov.insert(5, "dead", died, True) 



def check_covRes_icu_dead(df_cov):
    ''''
    #Patients that were tested for covid and were NEGATIVE:
    '''
    #if df_cov.icu == 1 AND df_cov.dead == 1 (i.e. they were imported to icu and they died), it is assumed a mistake occured at df_cov['covid_res']==2 and 'covid_res' is altered to 1
    df_cov.loc[(df_cov.icu == 1) & (df_cov.dead == 1), 'covid_res']=1
    #else if df_cov.icu == 1 OR df_cov.dead == 1, we could not assume were the error occured. In this case the record is removed
    df_cov.drop(df_cov[ (df_cov['covid_res']==2) &  ( (df_cov['icu']==1) | (df_cov['dead']==1)) ].index, inplace=True)

    
    
def dataframeToCsv(df, csvName):
    #convert dataframe to .csv
    df.to_csv( csvName+'_without_index.csv', sep=',',index=False)

    #index=False, the row index number (of the initial dataframe) is not written.
    df.to_csv( csvName+'.csv', sep=',')



def uniqueValues(df_name):
    df=pd.read_csv(df_name +'.csv', delimiter = ',')
    for col in df.columns:
        print("column name: ", col, " unique values: ")
        print(df[col].unique())
        

    
def change_1To0_2To1(columns, df_cov):
    for col in columns:
        df_cov.loc[(df_cov[col] == 1) , col]=0   #anapodi periptwsi
        df_cov.loc[(df_cov[col] == 2) , col]=1
        
        
def filters(df_cov, dataset, _1To0_2To1, extra):
    
    drop_icu_covidRes_Nan(df_cov, dataset, extra)

    '''
    Sex: Female=1, Male=2
    Pregnancy: Yes=1, No=2
    In 'sex' there were no missing values. If there's a male and 'pregnancy' is anything but 2, it is altered to 2
    '''

    #Making all men "not pregnant"
    df_cov.loc[(df_cov.sex == 2),'pregnancy']=2
    #df_cov['pregnancy']= np.where((df_cov.sex == 2), 2, df_cov.pregnancy)
    

    #Filtering all rows but 'sex', 'patient_type', 'entry_date', 'date_symptoms', 'date_died', 'age', 'covid_res', 'icu'
    #and replacing all not defined values, i.e. 97,98,99, with NA 
    but=['sex', 'patient_type', 'entry_date', 'date_symptoms', 'date_died', 'age', 'covid_res', 'icu']
    for attr in df_cov.columns:
        if attr not in but:
            df_cov.loc[ (df_cov[attr] == 97) | (df_cov[attr] == 98) | (df_cov[attr] == 99), attr ] = 3
            #df_cov.loc[ (df_cov[attr] == 97) | (df_cov[attr] == 98) | (df_cov[attr] == 99), attr ] = np.nan

    createAgeAttr(df_cov)
    createPhasesAttr(df_cov, dataset)


    date_cols=['first_phase','final_phase','total']
    clearPhaseAttr(df_cov,date_cols)
    createGroupPhaseAttr(df_cov,date_cols)

    createDeadAttr(df_cov)   
    check_covRes_icu_dead(df_cov)

    #uniqueValues("covid_cleared_without_index")      
    #uniqueValues("mexico_cleared_without_index")  
    print(len(df_cov))
    #print(df_cov.columns)
    
    if _1To0_2To1:
        columns_Yes_No=['sex', 'patient_type','dead', 'intubed', 'pneumonia', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension',
                    'other_disease', 'cardiovascular', 'obesity', 'renal_chronic','tobacco', 'contact_other_covid', 'covid_res', 'icu']
        change_1To0_2To1(columns_Yes_No, df_cov)

    
    
def main():
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    '''
    define dataset
    '''
    dataset='kaggle'
    #dataset='mex'
    _1To0_2To1 = 1
    
    #df_cov : data_frame_covid
    if dataset=='kaggle':
        df_cov=pd.read_csv('covid.csv', delimiter = ',')
        
        #I permanently remove/drop the column id from data_frame_covid, so I set inplace=True
        df_cov.drop(labels="id", axis=1, inplace=True)
        #print(df_cov.shape)
    
        filters(df_cov, dataset, _1To0_2To1, extra=0)
        dataframeToCsv(df_cov,'covid_cleared')
    else:

        for i in range(2):
             rmCols_nonCovid_translate_csv()
             df_cov=pd.read_csv('covid_Mexico.csv', delimiter = ',')
             filters(df_cov, dataset, _1To0_2To1, extra=i)
             if i:
                 dataframeToCsv(df_cov,'mexico_cleared_extra')
             else:
                 dataframeToCsv(df_cov,'mexico_cleared')
        
          
    '''
    df_cov.head() # data_frame_covid.head(x), x number of records
    df_cov.tail()
    df_cov.sample(5) # 5 in order to retrieve 5 random lines. 5 could be replaced with any number.
    '''

    


if __name__ == "__main__":
    main()