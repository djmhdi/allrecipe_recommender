# Recipe Project - Allrecipes.com Vegetarian Dish recommender
## Business Question for this Exercise/Project
* Build a recipe recommender using collaborative filtering from data scraped from the Allrecipes.com website.

---
## Data Understanding
* Data was collected from the allrecipe.com website, collecting usernames, ratings(1-5), recipe name, and the date that the rating was given.  The initial goal was to collect user/ratings information from all recipes but in the interest of time and to streamline the data collection, the project has now been changed into a vegetarian recipe recommender.  
---
## Data Preparation
* A web scraper algorithm was created to gather ~4000+ individual recipe weblinks from the allrecipe search page.  Once the weblink list was collected, a second scraping program collected the relevant information per recipe page - recipe name, user_name, user rating which varied per recipe.  For recipes with a large number of ratings, the number of user and rating combination was capped at 100, mainly to facilitate the completion of a minimum viable product. The distribution of review totals per recipe is shown. ![alt text](https://github.com/pineda-vv/allrecipe_recommender/blob/master/data/review_dist.png)

* The data collected by the scraper were stored in a mongod database.  Exploratory data analysis was performed using a Jupyter notebook where the mongod contents were transformed into a continuous table containing user_id, recipe_id, and rating.  Each user_id and recipe_id were integers that identify a unique user and recipe name respectively. The final dataset had ~162K unique reviews/ratings with ~4000 recipe_id and ~87.8K user_id.

### Snapshot of captured Data

* Exploratory data analysis (EDA)

* #### Figure 1 Ratings Distribution
*



---
## Modeling
### Using Pyspark and Alternating Least Squares
