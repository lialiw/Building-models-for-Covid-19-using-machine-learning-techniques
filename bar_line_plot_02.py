# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:20:29 2021

@author: user
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#def createBarPlots(col,norm,title,xlabel,ylabel,graph_label,imgName):
    
def createBarPlots(df_cov,attr,norm,title,xlabel,ylabel, xValues,imgName):  
    
    #opacity = 0.3
    bar_width = 0.2
    rot=0
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
        
    #plt.style.use('classic')
    
    plt.title(title)
    plt.xlabel(xlabel)

    
    if norm:
        plt.ylabel(ylabel + " Normalized")
        if isinstance(attr,list):
            '''
            #Step 1: Create a dataframe that stores the count of each non-zero class in the column counts
            count_df = df.groupby(['Symbol','Year']).size().reset_index(name='counts')
            '''
            count_df = df_cov.groupby([ attr[0],attr[1] ] ).size().reset_index(name='counts').divide(other = len(df_cov))
            
            '''
            #Step 2: Now use pivot_table to get the desired dataframe with counts for both existing and non-existing classes.
            count_df = df.groupby(['Symbol','Year']).size().reset_index(name='counts')
            '''            
            temp = pd.pivot_table(count_df, index= [ attr[0],attr[1] ], values='counts', fill_value = 0, dropna=False, aggfunc=np.sum)
            
            #Convert df_cov (DataFrame) to val (Series)
            val=temp.squeeze()
        else:
            val = df_cov[attr].value_counts(normalize=True, sort=False)
        
    else:
        plt.ylabel(ylabel)
        if isinstance(attr,list):
            #Step1
            count_df = df_cov.groupby([ attr[0],attr[1] ] ).size().reset_index(name='counts')
            #Step2
            temp = pd.pivot_table(count_df, index= [ attr[0],attr[1] ], values='counts', fill_value = 0, dropna=False, aggfunc=np.sum)
            #Convert df_cov (DataFrame) to val (Series)
            val=temp.squeeze()
        else:
            val = df_cov[attr].value_counts(sort=False)
    
    print(attr, val.index, val) 
    
    if len(xValues)>len(val):
        x_pos, val_withZeros =[],[]  
        for i in range(len(xValues)):
            
            if (i+1) in val:
                if isinstance(attr,list):
                    keys=val.keys()
                    for key in keys:
                        if i+1==key[0]:
                            val_withZeros.append(val[key])
                else:
                    val_withZeros.append(val[i+1])
                
     
            else:
                val_withZeros.append(0)
            x_pos.append(i)
            
            if len(val_withZeros)>len(xValues):
                val_withZeros.pop()

        #print(val_withZeros)    
        bar = plt.bar(x_pos, val_withZeros, bar_width, align='center', color='darkcyan')
    else:
        x_pos = [i for i, _ in enumerate(val.index)] 
        bar = plt.bar(x_pos, val, bar_width, align='center', color='darkcyan')
    '''
    print(attr)
    #print(x_pos)
    print(val)
    #print(len(xValues))
    '''
    plt.xticks(x_pos, xValues, rotation=rot)

    for rect in bar:
        height = rect.get_height()
        if norm:
            #('%f' % x).rstrip('0').rstrip('.')
            plt.text(rect.get_x() + rect.get_width()/2.0, height, ('%f' % float(height)).rstrip('0').rstrip('.'), ha='center', va='bottom')
        else:
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

    
    
    if norm:
        plt.savefig(imgName + "_normalized.png", dpi=250, transparent=True, bbox_inches='tight')
    else:
        plt.savefig(imgName + ".png", dpi=250, transparent=True, bbox_inches='tight')

    plt.close('all')
    plt.clf()
    


def  createLinePlot(df_cov,attr,title,xlabel,ylabel,imgName):  
    #opacity = 0.3
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    #plt.style.use('classic')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    val = df_cov.groupby([ attr ] )[ attr ].value_counts() 
  
    #print(type(val.index ) )
    #print(type(val))
    
    x_pos = [i for i, _ in enumerate(val.index)]

    plt.plot(x_pos, val, color='darkcyan')
    
    plt.savefig(imgName + ".png", dpi=250, transparent=True, bbox_inches='tight')
    plt.clf()


    
def draw_uniqueAttrPlots(df_cov,groupA,title_GroupA, xAxisVal_GroupA,groupB,title_GroupB, xAxisVal_GroupB, groupC2,title_GroupC2, xAxisVal_GroupC2, 
                         groupC1, title_GroupC1, xAxis_GroupC1,yAxis_GroupC1,groupC3,title_GroupC3,xAxisVal_GroupC3): 
    gA,gB,gC1,gC2,gC3= 0,0,0,0,0
    
    for attr in df_cov.columns:
        '''
        if attr in groupA:

            if attr=='pregnancy':
                women=df_cov[['sex','pregnancy']].copy()
                women.drop(women [ women ['sex']==2 ].index, inplace=True)
                createBarPlots(women,attr,True,title_GroupA[gA],"","", xAxisVal_GroupA[gA],attr)
                createBarPlots(women,attr,False,title_GroupA[gA],"","", xAxisVal_GroupA[gA],attr)
            else:
                createBarPlots(df_cov,attr,True,title_GroupA[gA],"","", xAxisVal_GroupA[gA],attr)
                createBarPlots(df_cov,attr,False,title_GroupA[gA],"","", xAxisVal_GroupA[gA],attr)
            gA+=1
        '''
        if  attr in groupB:
            if attr=='pneumonia':
                createBarPlots(df_cov,attr,True,title_GroupB[gB],"","",['Had=1', 'Did not have=2'], attr)
                createBarPlots(df_cov,attr,False,title_GroupB[gB],"","",['Had=1', 'Did not have=2'], attr)
            else:
                createBarPlots(df_cov,attr,True,title_GroupB[gB],"","",xAxisVal_GroupB, attr)
                createBarPlots(df_cov,attr,False,title_GroupB[gB],"","",xAxisVal_GroupB, attr)
            gB+=1

        '''
        if attr in groupC2:
            createBarPlots(df_cov,attr,True,title_GroupC2[gC2],"","",xAxisVal_GroupC2[gC2], attr)
            createBarPlots(df_cov,attr,False,title_GroupC2[gC2],"","",xAxisVal_GroupC2[gC2], attr)
            gC2+=1

        if attr in groupC1:
            createLinePlot(df_cov,attr,title_GroupC1[gC1],xAxis_GroupC1[gC1],yAxis_GroupC1[gC1], attr)
            gC1+=1

        if attr in groupC3:
            
            #Normalizarion in this sub df_cov, not in total
            
            df=df_cov[ [attr]].copy()
            df.drop(df[ (df[attr]>20) | (df[attr]==-1)].index, inplace=True)
            createBarPlots(df,attr,True,title_GroupC3[gC3],"","", xAxisVal_GroupC3, attr +'_20')
            createBarPlots(df,attr,False,title_GroupC3[gC3],"","", xAxisVal_GroupC3,attr +'_20')
            gC3+=1
        #'''
        
        
        
def draw_combinationAttrPlots(df_cov, divisors,title_divisors, group_Mul, title_group_Mul, indexes_xVal, excluded):  
    
    createBarPlots(df_cov,divisors,True,'ICU and Deaths',"","", indexes_xVal[0] ,'icu_dead')
    createBarPlots(df_cov,divisors,False,'ICU and Deaths',"","", indexes_xVal[0] ,'icu_dead')
    

    for i in range(len(divisors)):
        for j in range(len(group_Mul)):
            combination=[ group_Mul[j] , divisors[i] ]
            
            if group_Mul[j] in excluded:
                
                if (group_Mul[j] == 'sex') or group_Mul[j] == 'pneumonia' or group_Mul[j] == 'intubed':
                    createBarPlots(df_cov, combination, True, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[0], group_Mul[j]+"_"+divisors[i])
                    createBarPlots(df_cov, combination, False, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[0], group_Mul[j]+"_"+divisors[i])
             
                if group_Mul[j] =='pregnancy':
                    women=df_cov [ ['sex','pregnancy',divisors[i] ] ].copy()
                    women.drop(women [ women ['sex']==2 ].index, inplace=True)
                    createBarPlots(women, combination, True, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[1], group_Mul[j]+"_"+divisors[i])
                    createBarPlots(women, combination, False, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[1], group_Mul[j]+"_"+divisors[i])
              
                if group_Mul[j] == 'age_group':
                    createBarPlots(df_cov, combination, True, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[2], group_Mul[j]+"_"+divisors[i])
                    createBarPlots(df_cov, combination, False, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[2], group_Mul[j]+"_"+divisors[i])
             
                if group_Mul[j] == 'group_first_phase':
                    createBarPlots(df_cov, combination, True, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[3], group_Mul[j]+"_"+divisors[i])
                    createBarPlots(df_cov, combination, False, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[3], group_Mul[j]+"_"+divisors[i])
              
            else:
                createBarPlots(df_cov, combination, True, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[1], group_Mul[j]+"_"+divisors[i])
                createBarPlots(df_cov, combination, False, title_group_Mul[j]+title_divisors[i] ,"","", indexes_xVal[1], group_Mul[j]+"_"+divisors[i])



'''
Printing bar plots for all the attributes:
    'sex', 'patient_type', 'entry_date', 'date_symptoms', 'date_died','dead', 'intubed', 'pneumonia', 'age', 'age_group', 'pregnancy',
    'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension','other_disease', 'cardiovascular', 'obesity', 'renal_chronic','tobacco', 
    'contact_other_covid', 'covid_res', 'icu', 'first_phase','final_phase', 'total', 'group_first_phase', 'group_final_phase','group_total'
'''
def main():    
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)

    '''
    define dataset
    df_cov : data_frame_covid
    '''
    
    
    '''
    Correct:
    'Outpatient=1', 'Inpatient=2'
    but for convenience:
    'Inpatient=1', 'Outpatient=2'
    here
    '''
    dataset='kaggle'
    #dataset='mex'
    
    if dataset == 'kaggle':
        #df_cov=pd.read_csv('covid_cleared.csv', delimiter = ',')
        df_cov=pd.read_csv('covid_cleared_without_index.csv', delimiter = ',')
        xAxisVal_GroupA = [    ['Female=1', 'Male=2'], ['Inpatient=1', 'Outpatient=2'],['Died=1','Did not die=2'], ['Intubed=1', 'Not intubed=2'], 
                               ['[0-17]', '[18-39]', '[40-64]', 'age>=65'], ['Pregnant=1', 'Not pregnant=2', 'NA=3'], ['Was=1', 'Was not=2', 'NA=3'],
                               ['Smoked=1', 'Did not smoke=2', 'NA=3'],['Positive=1', 'Negative=2'], ['Imported=1', 'Not Imported=2']    ]
            
        xAxis_GroupC1 = ["Days, Zero day: 01/01/2020, End date: 29/06/2020","Days, Zero day: 01/01/2020, End date: 28/06/2020","Days, Zero day: 13/01/2020, End date: 29/06/2020",
                         "Number of days in first phase for each patient","Number of days in final phase for each patient","Total number of days for each patient"]
        
        excluded=['sex','pneumonia','intubed', 'pregnancy', 'age_group', 'group_first_phase']
    
    else:
        #df_cov=pd.read_csv('mexico_cleared_without_index.csv', delimiter = ',')
        df_cov=pd.read_csv('mexico_cleared_extra_without_index.csv', delimiter = ',')
        xAxisVal_GroupA = [    ['Female=1', 'Male=2'], ['Inpatient=1', 'Outpatient=2'],['Died=1','Did not die=2'], ['Intubed=1', 'Not intubed=2', 'NA=3'], 
                               ['[0-17]', '[18-39]', '[40-64]', 'age>=65'], ['Pregnant=1', 'Not pregnant=2', 'NA=3'], ['Was=1', 'Was not=2', 'NA=3'],
                               ['Smoked=1', 'Did not smoke=2', 'NA=3'],['Positive=1', 'Negative=2'], ['Imported=1', 'Not Imported=2']    ]
            
        xAxis_GroupC1 = ["Days, Zero day: 01/01/2020, End date: 22/03/2021","Days, Zero day: 01/01/2020, End date: 21/03/2021","Days, Zero day: 07/01/2020, End date: 22/03/2021",
                         "Number of days in first phase for each patient","Number of days in final phase for each patient","Total number of days for each patient"]
        excluded=['sex','pneumonia', 'pregnancy', 'age_group', 'group_first_phase']
    #print(df_cov.columns)
    
    
    groupA = ['sex', 'patient_type','dead','intubed','age_group','pregnancy','inmsupr','tobacco', 'covid_res', 'icu']

    title_GroupA = ["Sex","Patient's type","Deaths","Intubation","Age group","Pregnancy -only women are included-", "Immunosuppressed", "Tobacco","Covid test's result","Intensive Care Unit- ICU"]
    #-> no outpatients


    
    groupB = ['pneumonia','diabetes','copd', 'asthma','hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic','contact_other_covid']

    title_GroupB = ["Pneumonia", "Diabetes", "Chronic Obstructive Pulmonary Disease - COPD", "Asthma", "Hypertension", "Other disease", 
                    "Cardiovascular disease", "Obesity", "Renal Chronic - Chronic kidney disease", "Contacted other diagnosed with covid"]
    
    xAxisVal_GroupB = ['Had=1', 'Did not have=2', 'NA=3']
    
    

    groupC1 = ['entry_date', 'date_symptoms','date_died','first_phase','final_phase','total'] 

    title_GroupC1 = ["Patient's entry date at the hospital", "Patient's date of (first) symptoms", "Patient's date of death",
                     "Period: date of (first) symptoms - date of entry", "Period: date of entry - date of death", "Period: date of (first) symptoms - date of death"]
     
    yAxis_GroupC1 = ["Number of people who entered the hospital per day", "Number of people who displayed their (first) symptoms per day", "Number of people who died per day","","",""]   


    
    groupC2 = ['group_first_phase', 'group_final_phase','group_total']

    title_GroupC2 = ["Groups of periods of first phase", "Groups of periods of final phase", "Groups of periods of total time"]

    xAxisVal_GroupC2 = [   ['0:0', '1:<10', '2:<20', '3:<30', '4:<40', '5:<50', '7:<70', '8:>=70'], 
                        [ '0:0', '1:<10', '2:<20', '3:<30', '4:<40', '5:<50','7:<70', '8:>=70', '-1:not dead',],
                        [ '0:0', '1:<10', '2:<20', '3:<30', '4:<40', '5:<50','7:<70', '8:>=70','-1:not dead'] ]     



    groupC3 = ['first_phase', 'final_phase','total']

    title_GroupC3 = ["Cases in which first phase lasted <=20 days", "Cases in which final phase lasted <=20 days", "Cases in which total number of days was less than 20 days"]
    xAxisVal_GroupC3 =['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']

    '''    
    #Call function    
    draw_uniqueAttrPlots(df_cov,groupA,title_GroupA, xAxisVal_GroupA,groupB,title_GroupB, xAxisVal_GroupB, groupC2,title_GroupC2, xAxisVal_GroupC2,
                         groupC1, title_GroupC1, xAxis_GroupC1,yAxis_GroupC1,groupC3,title_GroupC3,xAxisVal_GroupC3)
    #'''


    '''
    In group_Mul:
        indexes : '1-1','1-2','2-1','2-2' are going to be used to describe bar plots
        for age_group -> '[0-17]:1', '[18-39]:1', '[40-64]:1', 'age>=65:1'     
        for 'group_first_phase' -> '0:1', '1:1', '2:1', '3:1', '4:1', '5:1', '7:1', '8:1', '0:2', '1:2', '2:2', '3:2', '4:2', '5:2', '7:2', '8:2'  
    '''

    divisors = ['icu','dead'] 
    title_divisors = ['ICU', 'Death']
    
    indexes_xVal = [    ['1-1','1-2','2-1','2-2'], ['1-1','1-2','2-1','2-2','3-1','3-2'], 
                    ['[0-17]:1', '[0-17]:2', '[18-39]:1', '[18-39]:2', '[40-64]:1', '[40-64]:2', 'age>=65:1', 'age>=65:2' ],
                    ['0:1', '0:2', '1:1', '1:2', '2:1', '2:2', '3:1', '3:2', '4:1', '4:2', '5:1', '5:2', '7:1', '7:2', '8:1', '8:2']  ]


    group_Mul = ['sex', 'intubed', 'pneumonia', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension',
                 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic','tobacco','age_group', 'group_first_phase']
    
    title_group_Mul = ["Sex and ","Intubation and ","Pneumonia and ","Pregnancy and ","Diabetes and ", "COPD and ","Ashtma and ","Immunosuppressed and ","Hypertension and ",
                       "Other disease and ","Cardiovascular disease and ","Obesity and ", "Renal chronic and ", "Smoking and ", "Age group and ", "First phase's group and "]
       
    #Call function
    draw_combinationAttrPlots(df_cov, divisors,title_divisors, group_Mul, title_group_Mul, indexes_xVal, excluded)
    #'''
    
    #createBarPlots(df_cov,['pneumonia','dead'],True,'Pneumonia and deaths',"","", ['1-1','1-2','2-1','2-2','3-1','3-2'],"pneum_dead")
    #createBarPlots(df_cov,'intubed',False,'Intubation',"","", ['Intubed=1', 'Not intubed=2'],'intubed')  
    #createBarPlots(df_cov,'intubed',True,'Intubation',"","", ['Intubed=1', 'Not intubed=2'],'intubed')    
  
    
  
    
if __name__ == "__main__":
    main()