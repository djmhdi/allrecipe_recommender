from recipe_recommender import RecipeRecommender
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main(filename):
    df = pd.read_csv(filename)
    working = df[['user_id', 'recipe_id', 'rating']]
    data_prep(working)

def data_prep(df):
    y = df['rating'].values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)
    cols = 'user_id recipe_id rating'.split()
    train_df.columns = cols
    test_df.columns = cols
    test_df_input = test_df[['user_id', 'recipe_id']]
    optimization(train_df, test_df)

def optimization(train_df, test_df):
    params = {'rank': [1, 2, 4, 6, 8, 10],
              'regParam':[0.01, 0.1, 0.2, 0.5, 0.8]
             }






if __name__ == '__main__':
    filename = '../data/all_recipe_clean.csv'
    main(filename)
