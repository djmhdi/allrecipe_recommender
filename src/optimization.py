from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error
from recipe_recommender import RecipeRecommender
import numpy as np
import pandas as pd


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
    test_df_preds = test_df[['user_id', 'recipe_id']]
    opt_list = optimization(train_df, test_df_preds, test_df)



def optimization(train_df, test_df_preds, test_df):
    output = []
    params = {'rank':[1, 2, 5, 10, 50, 250], 'reg':[0.01, 0.1, 0.5, 1]}
    combi = list(ParameterGrid(params))
    for dct in combi:
        rank, reg = dct['rank'], dct['reg']
        model = RecipeRecommender(rank, reg)
        model.fit(train_df)
        preds = model.transform(test_df_preds)
        rmse_score = rmse(preds, train_df, test_df)
        output.append([rank, reg, rmse_score])
    return output

def rmse(preds, train_df, test_df):
    train_rating_mean = train_df['rating'].mean()
    new_df = preds.merge(test_df, how='left', on=['user_id', 'recipe_id'])
    new_df['user_bias'] = new_df['user_id'].apply(user_bias)
    new_df['recipe_bias'] = new_df['recipe_id'].apply(recipe_bias)
    new_df['total_bias'] = new_df['recipe_bias'] + new_df['user_bias']
    new_df['adjusted'] = new_df['prediction'] + new_df['total_bias']
    new_df['adjusted'].fillna(new_df['total_bias'].apply(lambda x: train_rating_mean + x ), inplace=True)
    y_pred = new_df['adjusted']
    y_true = new_df['rating']
    return np.sqrt(mean_squared_error(y_true, y_pred))

def user_bias(col):
    user_groups = train_df['rating'].mean() - train_df.groupby('user_id')['rating'].mean()
    if col in user_groups.index:
        return user_groups[col]
    else:
        return 0

def recipe_bias(col):
    recipe_groups = train_df['rating'].mean() - train_df.groupby('recipe_id')['rating'].mean()
    if col in recipe_groups:
        return recipe_groups[col]
    else:
        return 0

if __name__ == '__main__':
    filename = '../data/all_recipe_clean.csv'
    main(filename)
