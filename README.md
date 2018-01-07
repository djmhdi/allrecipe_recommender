# Recipe Project - Vegetarian Dish recommender
## Business Question for this Exercise/Project
* Build a recipe recommender system (RS) from data scraped from the allrecipes.com [website](allrecipes.com).

---
## Data Understanding
* To build an RS using collaborative filtering, users, items(recipes), and the rating that each user gives for an item are required.  The allrecipe.com star rating system ranges from 1-star(not-highly rated) to the full 5-star(highly rated).   There is a huge amount of information that can be collected from this website and the initial goal was to collect user/recipe/ratings information from every recipe but in the interest of time and to streamline the data collection, the project was streamlined into an RS for vegetarian recipes.  
---
## Data Preparation
* A web scraper algorithm was created to gather ~4000+ individual recipe weblinks from the allrecipe search page.  Once the weblink list was collected, a second scraping program accessed each weblink to extract the relevant information - recipe name, user_name, user rating - which varied per recipe.  For recipes with a large number of ratings, the number of user and rating combination was capped at 100, mainly to facilitate the completion of a minimum viable product. For recipes that have not been rated by any user, a rating score of 0 was assigned. The distribution of total reviews per recipe is shown. ![alt text](https://github.com/pineda-vv/allrecipe_recommender/blob/master/data/review_dist.png)

* The data collected by the scraper program were dumped in mongod.  Exploratory data analysis was performed in a Jupyter notebook where the mongod contents were transformed into a continuous table containing user_id, recipe_id, and rating.  Each user_id and recipe_id is an integer that corresponds to a unique user or recipe name. The pickled name:id dictionaries are in the [data](https://github.com/pineda-vv/allrecipe_recommender/tree/master/data) folder. The final dataset had ~162K unique reviews/ratings with ~4000 recipe_id's and ~87.8K user_id's.

---
## Snapshot of captured Data

* #### **Ratings Distribution** - This plot shows the overabundance of 5-star ratings compared to all the other possible scores.  
![alt text](https://github.com/pineda-vv/allrecipe_recommender/blob/master/data/ratings_dist.png)

* #### **Total count per rating category**
| **Rating** | **Counts** |
|:---:|:---:|
| **5** | **106922** |
| **4** | **39092** |
| **3** | **10151** |
| **2** | **3731** |
| **1** | **2751** |
| **0** | **24** |


---
## **Modeling**
#### Collaborative Filtering using Spark's Alternating Least Squares (ALS) - The user/item relation from explicit ratings (1-5) from 87786 users and 3913 recipes were analyzed using the alternating least squares (ALS) model from Spark MLlib.  The data was split into 80/20 training/test sets with the
