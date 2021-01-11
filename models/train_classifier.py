# import libraries
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
import pickle

def load_data(database_filepath):
    """
    Loads data from database filepath and splits into X, y and category_names
    Args:
    database_filepath str: Filepath to database containing training data
    Returns:
    X Series: DataFrame containing the messages (features)
    y DataFrame: DataFrame containig the data category labels (targets)
    category_names str: Response category names
    """
    # 1. Load the dataset
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'Messages'
    df = pd.read_sql_table(table_name,engine)
    # 2. Clean the dataset
    ## Drop child alone as only zeros
    df = df.drop(['child_alone'],axis=1)
    ## Map value 2 to value 1 (majority value)
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    # 3. Split dataset into X and y
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    # clean_tokens list: Processed text after case normalizing, tokenizing and lemmatizing
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # model_pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
         ('clf',MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])
    # hyper-parameter grid
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0)}

    # create model
    model = GridSearchCV(estimator=pipeline,
                        param_grid=parameters,
                        verbose=3,
                        cv=3)

    # cv = GridSearchCV(pipeline, param_grid=parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Show model's performance on test data
    Args:
    model: Trained model for evaluation
    X_test Series: Test features
    Y_test DataFrame: Test targets
    category_names: Target labels
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred,columns=Y_test.columns)
    for category in category_names:
        print(category)
        print(classification_report(Y_test[category],Y_pred_df[category]))


def save_model(model, model_filepath):
    """
    Saves trained model
    Args:
    model model: Model to save
    model_filepath str: Filepath to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
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

# Run
if __name__ == '__main__':
    main()
