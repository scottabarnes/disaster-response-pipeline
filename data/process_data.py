
import sys
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges data
    Args:
    messages_filepath str: Filepath to messages dataset
    categories_filepath str: Filepath to categories dataset
    Returns:
    df DataFrame: Joined dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='inner',on='id')
    return df

def clean_data(df):
    """
    Cleans raw input data for model training
    Args:
    df DataFrame: joined dataframe for cleaning
    Returns:
    df DataFrame: cleaned dataframe for model training
    """    # 1. Create category col df with values
    categories = df['categories'].str.split(';',expand=True)
    category_colnames = [e[:-2] for e in list(categories.iloc[1])]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        categories.fillna(0,inplace=True)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # 2. Merge category df back on to original DataFrame
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    # 3. Remove non binary values
    for category in category_colnames:
        df = df[(df[category] == 1) | (df[category] ==0)]
    return df

def save_data(df, database_filepath):
    """
    Saves cleaned dataset to database for subsequent loading
    Args:
    df DataFrame: Cleaned dataframe to be saved down
    database_filename str: Filepath to save database
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

# run
if __name__ == '__main__':
    main()
