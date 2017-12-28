# Recipe Project - Allrecipes.com Vegetarian Dish recommender
## Question for this Exercise/Project
* Build a recipe recommender using collaborative filtering from data scraped from the Allrecipes.com website.

---
## Data Understanding
* Data was collected from the allrecipe website, with the initial goal of collecting usernames, ratings(1-5), recipe name, and the date that the rating was given.  The initial goal was to collect as many recipes as possible but in the interest of time and to streamline the data collection, the project is now officially a vegetarian recipe recommender.  
---
## Data Preparation
* A web scraper algorithm was created to gather ~4000+ individual recipe weblinks from the allrecipe search page.  Once the weblink list was collected, a second scraping program collected the relevant information per recipe page - recipe name, user_name, user rating which varied per recipe.  For recipes with a large number of ratings, the number of user and rating combination was capped at 100, mainly to facilitate the completion of a minimum viable product.  

* The data collected by the scraper were stored in a mongod database.  Exploratory data analysis was performed on a Jupyter notebook where the mongod contents were transformed into a continuous table containing user_id, recipe_id, and rating.  Each user_id and recipe_id were integers that identify a unique user and recipe name respectively.

### Snapshot of captured Data

* Pooled recipes exploratory data analysis (EDA)
-- The histogram for the count of 'likes' for each unique recipe (recipes with 1, 2, 3....n) appear to show a Poisson distribution.
 ![alt text](https://github.com/pineda-vv/Data-Science-Projects/blob/master/recipe_project/data/latex_poisson_pmf.png)

* #### Figure 1 Ratings Distribution
* A) All recipes and categories
 ![alt text](https://github.com/pineda-vv/Data-Science-Projects/blob/master/recipe_project/data/distribution.png)
* B) Ratings Distribution for each category -- the 'vegetarian' category most likely include dessert, side dish, and vegetable main dish recipes.
![alt text](https://github.com/pineda-vv/Data-Science-Projects/blob/master/recipe_project/data/distribution_ingredients.png)

---
## Modeling Part 1
#### Popularity - Using Gradient Boosting Classifier
1.  Initial modeling centered on trying to build a simple predictive model on whether a recipe (~9300 collected) is "popular" or not.  Intuitively, the recipes with a rating of either 1 or 0 could have been used as the positive ('not popular') class.  However, some of these recipes are likely newly uploaded to the site and perhaps have not been seen/rated by enough viewers.  Thus, a threshold was chosen instead. Recipes with less than or equal to 10 likes were labeled as the positive class (Figure 2A)  This modeling worked well, after engineering some features based on the text of the recipes.  

2. Model Evaluation -- cross-validation metrics of popularity classifier.

* #### Figure 2
![alt text](https://github.com/pineda-vv/Data-Science-Projects/blob/master/recipe_project/data/classifier_analysis.png)

## Modeling Part 2
#### Clustering of Recipes using non-negative matrix factorization (NMF) and t-distributed stochastic neighbor embeding (t-SNE)
1. Non-negative matrix factorization used to extract the top topics/word groups in the recipe text (ingredients) as well as the title.

![alt text](https://github.com/pineda-vv/Data-Science-Projects/blob/master/recipe_project/data/recipe_text_tsne.png)
