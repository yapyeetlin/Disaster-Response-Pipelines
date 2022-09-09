import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Get current working directory
    cwd = os.getcwd()

    # Import both "messages" and "categories" file from csv
    messages = pd.read_csv(os.path.join(cwd,messages_filepath))
    categories = pd.read_csv(os.path.join(cwd,categories_filepath))

    # merge both csv-files into DataFrame and return 
    return pd.merge(messages, categories, on="id")


def clean_data(df):
    # Delimit column "categories" into multiple columns and rename column names
    categories = df["categories"].str.split(";",expand=True)
    categories.columns = categories.iloc[0].str.split("-", expand=True)[0]
    
    # Convert all values in the delimited columns into int-type 
    categories = categories.applymap(lambda x: int(x.split("-")[-1]))

    # drop the original categories column from `df`
    df = df.drop(columns="categories")

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    # Add-on: replace '2' in column "related" with '1'
    df["related"] = df["related"].replace(2,1)
    
    return df


def save_data(df, database_filename):
    # Save the clean dataset into an sqlite database
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("DisasterResponse", engine, index=False)


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


if __name__ == '__main__':
    main()