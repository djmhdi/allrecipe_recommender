import pyspark as ps
#from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import logging
import pandas as pd
import numpy as np


class RecipeRecommender():


    def __init__(self):
        """Constructs a RecipeRecommender"""
        self.logger = logging.getLogger('reco-cs')


    def fit(self, ratings, rank, regParam):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 3)
                  with columns 'user', 'recipe', 'rating'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        spark = (
            ps.sql.SparkSession.builder
            .master('local[4]')
            .appName('BVS')
            .getOrCreate()
        )

        sc = spark.sparkContext
        # Convert a Pandas DF to a Spark DF
        ratings_df = spark.createDataFrame(ratings)

        # print ratings_df.show()
        # print ratings_df.printSchema()

        als_model = ALS(
            maxIter=5,
            rank=rank,
            itemCol='recipe_id',
            userCol='user_id',
            ratingCol='rating',
            nonnegative=True,
            regParam=regParam,
            coldStartStrategy="drop"
            )

        # Train the ALS model. We'll call the trained model `recommender`.
        self.recommender_ = als_model.fit(ratings_df)
        self.logger.debug("finishing fit")
        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.
        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user_id', 'recipe_id'
        Returns
        ----------
        dataframe : a pandas dataframe with columns 'user_id', 'recipe_id', 'prediction'
                    column 'prediction' containing the predicted rating
        """
        spark = (
            ps.sql.SparkSession.builder
            .master('local[4]')
            .appName('BVS')
            .getOrCreate()
        )
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        # requests['rating'] = np.random.choice(range(1, 5), requests.shape[0])
        # Convert a Pandas DF to a Spark DF
        requests_df = spark.createDataFrame(requests)
        self.predictions = self.recommender_.transform(requests_df)
        self.logger.debug("finishing predict")
        return(self.predictions.toPandas())

    def evaluate(self, requests):
        """
        Input - dataframe with rating and prediction for each user/recipe in
        the test set.
        Out - root mean squared error for the test set
        """
        spark = (
            ps.sql.SparkSession.builder
            .master('local[4]')
            .appName('BVS')
            .getOrCreate()
        )
        requests_df = spark.createDataFrame(requests)
        evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            )

        rmse = evaluator.evaluate(requests_df)
        return (rmse)

    def recommend_for_all(self, who='users', number=10):
        """
        Returns a dataframe for recommendations for each user or for each recipe
        """
        if who == 'users':
            user_recs = self.recommender_.recommendForAllUsers(number)
            return (user_recs.toPandas())
        else:
            recipe_recs = self.recommender_.recommendForAllItems(number)
            return (recipe_recs.toPandas)

if __name__ == "__main__":
    # logger = logging.getLogger('reco-cs')
    # logger.critical('you should use run.py instead')
    pass
