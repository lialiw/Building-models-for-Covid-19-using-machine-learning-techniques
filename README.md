# COVID-19 Predictive Modeling Repository

## Overview
This repository contains Python scripts designed for predictive modeling to analyze the progression of COVID-19 in patients. It aims to provide insights into critical outcomes such as ICU admissions and patient mortality. The models leverage machine learning techniques applied to datasets obtained from Kaggle and the Mexican government's Open Data General Directorate of Epidemiology.

## Repository Structure
- `extractData_02.py`: Preprocesses the initial datasets.
- `sampling_02.py`: Splits the datasets into training, testing and validation sets.
- `bar_line_plot_02.py`: Generates descriptive plots for the pre-processed datasets.
- `logReg_02.py`: Implements logistic regression analysis.
- `pca_LogReg.py`: Applies logistic regression on PCA-transformed data.
- `decTree_02.py`: Develops decision tree models.
- `decTree_paral.py`, `decTree_paralPool.py`: Parallel implementations of decision trees.
-  `randomForest_02.py`: Random Forest models.
- `kMeans_01.py`: Conducts k-means clustering on pre-processed datasets.
- `pca_01.py`: Performs principal component analysis.

## Project Objective
The primary aim of this project is to develop three different predictive models concerning the evolution of COVID-19 in patients:
1. **ICU Admission Prediction**: Determines the likelihood of a patient being admitted to the ICU.
2. **Mortality Prediction (with ICU data)**: Predicts the likelihood of patient mortality, considering ICU admission as an independent variable.
3. **Mortality Prediction (without ICU data)**: Forecasts patient mortality without considering ICU admission status.

Code was developped, while conducting research for my bachelor degree's thesis: "Building models for Covid-19 using machine learning techniques" 

## Models and Techniques
The models employ various machine learning methods including Logistic Regression, Decision Trees, Random Forest, k-Means Clustering and Principal Components Analysis (PCA). These models are intended to help alleviate the pressures on healthcare systems by providing real-time predictions which can enhance decision-making processes.

## Data Description
Data utilized can be found on my private repository [Covid19-Dataset](https://github.com/lialiw/Covid19-Dataset)
- **Kaggle Dataset**: "COVID-19 patient pre-condition dataset" consisting of clinical features and pre-conditions of the patients.
- **Mexican Government Dataset**: Patient records with similar clinical features, sourced from the Open Data General Directorate of Epidemiology.

## Potential Impact
The insights from these models could significantly impact healthcare resource management, helping in the strategic allocation of medical resources and personnel.

## Privacy Notice
All data used and provided in this repository respects patient confidentiality and is used strictly for academic and non-commercial purposes.
