import os 
import pandas as pd
from sklearn.impute import SimpleImputer 
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder as OHE


#load data for both testing and training 
titanic_data_train = pd.read_csv('train.csv')
titanic_data_test = pd.read_csv('test.csv')

#Need to make sure the data is clean for modeling.  Address categorical 
#variables as well as missing values. open notebook to test

#lets begin with the training data. First, isolate the base truth 

y = titanic_data_train['Survived']

#now drop this column from a copy of the trainging data 
#drop other useless columns, passenger ID
X_train = titanic_data_train
X_train.drop(['Survived'], axis = 1)
X_train.drop(['PassengerId'],axis =1)

#ok now i neeed to determine which data fields must be imputed 
#use notebook to check for NaN
#looks like cabin is mostly NaN, as well as age and a few embarked 

#question, as i clean my data do i want to clean the test data too?
# i can make a data pipeline and then feed whatever i need into it. 

def my_data_pipeline(dataframe):

    dataframe.drop(['Survived'], axis = 1)
    dataframe.drop(['PassengerId'],axis =1)
    dataframe['Sex'].replace(['male','female'],[0,1], inplace = True)
    #sex turned into binary.  
    #next i will imput the values 
    dataframe['Age'].fillna(dataframe['Age'].mean(),inplace = True)
    #for cabins, if they have a cabin = 1 no cabin = 0
    dataframe['Cabin'] = np.where(dataframe['Cabin'].isnull(),0,1)
    #strip the name to just be the title 
    dataframe['Title'] = dataframe['Name'].str.split(', ',expand = True)[1].str.split('.',expand = True)[0]
    dataframe.drop(['Name'],axis =1)
    