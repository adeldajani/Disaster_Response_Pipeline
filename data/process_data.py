#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''
This file contatins the functions to Extract the data , Transform it, and then Load it.

'''

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_path, categories_path):
    '''
    This Function Loads the data from teh specified file path.
    argument: 
    - messages_path: the file path to messages data.
    - categories_path: the file path to categories data.
    
    output:
    - df: the compined datadets 
    
    '''
    # load messages data
    messages = pd.read_csv(messages_path)
    
    #load categories data
    categories = pd.read_csv(categories_path)
    
    #merge the two datasets together
    df = pd.merge(messages, categories)
    
    
    
    return df

def clean_data(df):
    '''
    this function cleans the data.
    
    arguments:
    df: dataframe to be cleaned.
    
    outputs:
    df: cleaned input data contained the merged dataframes (messages) and (categories).
    
    '''
    
    #split categories into split category columns
    categories = df['categories'].str.split(expand = True, pat = ";")
    
    # select the first row of the categories dataframe
    row = categories.iloc[[0]]
    category_colnames = [category_names.split('-')[0] for category_names in row.values[0]]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    
    #Convert category values to just numbers 0 or 1.
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    df = df[df['related'] != '2']

    return df

def save_data(df, filename):
    '''
    This function saves teh data to an SQL Database
    
    Arguments:
    - df: Dataframe to be saved.
    -filename: the file name of the SQL Database
    
    '''
    engine = create_engine('sqlite:///' + filename)
    DB_name = filename[:len(filename) - 3]
    df.to_sql(DB_name, engine, index=False, if_exists = 'replace') 
    
def main():
    '''
    This function excute the ETL Pipeline
    
    '''
    if len(sys.argv) == 4:

        messages_path, categories_path, filename = sys.argv[1:]
        
        print('Extracting Data...\n Extracting Messages{} \n Extracting Categories{}'.format(messages_path, categories_path))
        
        df = load_data(messages_path, categories_path)
        
        print('Transforming data...\n')
        df = clean_data(df)
        
        print('Loading Data into the {} database ...\n'.format(filename))
        save_data(df, filename)
    else:
        print('ERROR...\n PLEASE ENTER THE CORRECT ARGUMENTS\n ')
        print('provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




