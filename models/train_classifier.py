import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('wordnet')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
sys.modules['sklearn.externals.joblib'] = joblib

def load_data(database_filepath):
    '''
    load_data
    Load data from SQL database and returns data as X(features), Y(labels) and category names

    Input:
    database_filepath   -> File path for SQL database

    Output:
    X                   -> Features 
    Y                   -> Labels
    category_names      -> Name for each label
    '''
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('DisasterResponse', engine)  
    
    X = df["message"]
    Y = df.iloc[:,-36:]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    tokenize
    Tokenize input text into smaller units and lemmatize the different inflected forms of a word so they can be analyzed as a single item

    Input:
    text                -> Message as text

    Output:
    clean_tokens        -> Small units of words after tokenization and lemmatization
    '''    

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    tokens = tokenizer.tokenize(text)

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    Build a multioutput classifier model with GridSearchCV. The steps of building a model are wrapped in a pipeline.

    Input:
    None

    Output:
    cv              -> Instance of GridSearchCV 
    '''    

    pipeline =  Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
                ])
    
    parameters = {'clf__estimator__max_depth': [5,10,20],
                'clf__estimator__min_samples_leaf': [10,20,50]}
    
    cv = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Evaluate the performance of best model found in GridSearchCV using test dataset imported from SQL database 

    Input:
    model           -> best model found in GridSearchCV
    X_test          -> Features of train dataset
    Y_test          -> Labels of train dataset
    category_names  -> Name for each label

    Output:
    None
    ''' 
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)

    print(classification_report(Y_test.values, Y_pred.values, target_names=category_names))


def save_model(model, model_filepath):
    '''
    save_model
    Save best model found in GridSearchCV in a given file path 

    Input:
    model           -> best model found in GridSearchCV
    model_filepath  -> location where the model is saved

    Output:
    None
    ''' 
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...(this might take up to 10 minutes)')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()