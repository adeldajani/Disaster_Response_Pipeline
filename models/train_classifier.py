#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''
this file contains the ML Pipeline for classifying desastir response messages.
to use this script successfully please enter the file path to the database, and then
the path where you would like to save the pickle file. 

'''


# import libraries
import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin

def load_db(db_filepath):
    '''
    this function loads the data from the database.
    
    inputs: db_filepath: the file path where the database is stored
    outputs: X:input features
             Y:output features
             cat_names: a list of category names
    
    '''
    engine = create_engine('sqlite:///' + db_filepath)
    DB_name = db_filepath[:len(db_filepath) - 3]
    df = pd.read_sql_table(DB_name, engine)
    X = df['message']
    Y = df[df.columns[4:]]
    cat_names = Y.columns
    return X, Y, cat_names

def tokenize(text):
    '''
    this function tokenises the text data and stores it in a list
    inputs:
        text : text messages data
    outputs:
        clean_tokens: a list containing the clean tokens 
    
    
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(:%[0-9a-fA-F][0-9a-fA-F]))'
    detect_urls = re.findall(url_regex, text)
    for url in detect_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tokens.append(lemmatizer.lemmatize(tok).lower().strip())
 
    return clean_tokens


def model_pipeline():
    '''
    this function builds a model pipeline and fits the model to the training data
    inputs:
        None
    outputs:
        model: fitted model.
    
    '''
    
    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 5)))
        ])
    parameters = {'clf__estimator__n_estimators': [30, 35, 40],
    'clf__estimator__min_samples_split': [5, 7, 9]
    
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)

    return model

def eval_model(model, cat_names, X_test, y_test):
    '''
    this function evaluates your model by printing out its performance.
    
    inputs:
        model: model pipeline.
        cat_names: category names
        X_test: input test data
        y_test: output test data
    output: 
        prints out the evaluation report.
    
    '''
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test.values, y_pred,
                            target_names=cat_names))
def save_model(model, pickle_path):
    '''
    this function saves the model to a pickle file.
    
    inputs:
        model: model pipeline
        pickle_path: the path where you want to save the model as a pickle file.
        
    '''
    with open(pickle_path, 'wb') as file:
        pickle.dump(model, file)
    
def main():
    if len(sys.argv) == 3:
        database_filepath, pickle_path = sys.argv[1:]
        
        print('loading data base .... \n  ')
        X, Y, cat_names = load_db(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X,Y)
        
        print('Building model.... \n')
        model = model_pipeline()
        
        print('Fitting the pipeline....\n')
        model.fit(X_train, y_train)
        
        print('Evaluating model.... \n')
        eval_model(model, cat_names, X_test, y_test)
        
        
        print('Saving model.... \n')
        save_model(model, pickle_path)
        
        print('Model Saved Successfully \n')
    else:
        print('input error please enter the the file path to the database first, then the path where you would like to save the pickle file')

        
if __name__ == '__main__':
    try:
        main()
    except:
        print('input error please enter the the file path to the database first, then the path where you would like to save the pickle file')


# In[ ]:




