---
title: "Site Energy Intensity Prediction - WiDS 2022 Hackathon"
description: "My top 10% Solution Approach for Site Energy Intensity Prediction Project which was part of the WiDS 2022 Kaggle Hackathon"
dateString: July 2022
draft: false
tags: ["ML", "Python", "Projects", "Climate Change AI"]
weight: 101
cover:
    image: "/blog/site-energy-intensity-pred/cover.jpeg"
---

# Overview

## Abstract

Climate change is a globally relevant, urgent, and multi-faceted issue heavily impacted by energy policy and infrastructure. Addressing climate change involves mitigation (i.e. mitigating greenhouse gas emissions) and adaptation (i.e. preparing for unavoidable consequences). Mitigation of GHG emissions requires changes to electricity systems, transportation, buildings, industry, and land use.

the lifecycle of buildings from construction to demolition were responsible for 37% of global energy-related and process-related CO2 emissions in 2020. Yet it is possible to drastically reduce the energy consumption of buildings by a combination of easy-to-implement fixes and state-of-the-art strategies.

## Dataset

The WiDS Datathon dataset was created in collaboration with Climate Change AI (CCAI) and Lawrence Berkeley National Laboratory (Berkeley Lab). WiDS Datathon participants will analyze differences in building energy efficiency, creating models to predict building energy consumption. Participants will use a dataset consisting of variables that describe building characteristics and climate and weather variables for the regions in which the buildings are located. Accurate predictions of energy consumption can help policymakers target retrofitting efforts to maximize emissions reductions.

## Brief Description of the Solution 

> The project involves predicting a buildings Site Energy Usage Intensity metric, which was the `site_eui` feature in the dataset (regression problem).  To solve this problem, I first separated the dataset into twelve individual datasets based on buildings with similar `site_eui` usage patterns and other characteristics.  I then engineer features, perform leave one group out cross validation, and finally train an ensemble model (XGBoost, LightGBM, and CatBoost regressors) for each dataset on the most powerful features.  My final solution ended up in the top 10% of the final leaderboard.

# Helper Functions

To make the code in the projects modular, I have used helper functions that include reusable code for **reading data**, **preprocessing tasks**, **feature engineering**, **cross validation** and **modelling**. Feel free to check out my [github](https://github.com/vedanthv/Site-Energy-Intensity-Prediction/blob/master/site-eui-pred-final.ipynb) repository for indepth docstrings of each function and post an [issue](https://github.com/vedanthv/Site-Energy-Intensity-Prediction/issues) if you want to report a bug for any of the functions.

# Data Preprocessing

## Dealing with Duplicate Data

* We see that there are 39 duplicated buildings in the train set and 5 duplicate buildings in the test set. 

* Since the number of duplicates is very less compared to the 75k samples in the dataset, I have removed the duplicates.

* I have left the duplicates in the test set as otherwise, it could affect the predictions.

# Feature Engineering

## Create Individual Datasets Based on Facility Types

In this section I split the datasets into 12 individual ones based on facility types with similar `site_eui` characteristics.

**Why I did this:**
- In this project, facility type was a feature that described what kind of facility a building was, and there were 60 total types.

- When I was doing exploratory data analysis, I noticed that there were an uneven number of facilities of different types.

- For example, there were 40k Multifamily buildings in the dataset, while other facility types such as Industrial only had a few hundred (or even less).

- Further, the distribution of site energy usage for different facility types was drastically different than others, and certain feature distributions were different as well.

- Therefore, it intuitively did not make sense to me to just train one machine learning model, since for example, how could a model trained with a significant portion of the data being related to Multifamily homes, make accurate predictions on other types where there was only 100 examples?

- So my idea behind this was to create smaller datasets of similar buildings that had similar energy usage patterns and feature distributions, and then to use these to train individual machine learning models on each dataset, with the ultimate goal of getting more accurate predictions overall in the end.

**How I did it:**
- In developing the individual datasets I tried a bunch of things, including:
    1. Naively grouping buildings based on the first word in their facility type, for example the types `Food_Sales` and `Food_Sales_Other` were grouped together in a dataset called `Food`.
    2. Using KMeans clustering to cluster similar buildings together, and then group the buildings based on cluster labels.
    3. Manually grouping buildings based on exploratory data analysis and iterating to optmize the Kaggle score

**What worked best:**

- In my final solution, I separate the train and test datasets into 12 individual data sets (each).
- You can see the exact groups of facility types that I used by reading the `get_manual_facility_groups` function in the Helper Sections [2nd Section] of the notebook 

# Missing Data

In this section I identify what data is missing in the dataset:

- We'll see that features related to fog or wind, have over 50% of the data missing in both the train and test set, and due to this, I don't use these features in my final solution.

- We'll also see that there is missing data in energy star rating and year built, which I deal with in the next section of this blog.

In this section, I impute the missing data in the `energy_star_rating` and `year_built` features:
- I use Ridge regression to impute the missing values, as I found this gave me the best results in terms of optimizing the final score of my solution.

- I also tried using XGBoost and LightGBM models to impute, but these did not do as well.
- First, I create a sklearn `ColumnTransformer` which one hot encodes the categorical features I used for imputation, as well as removes features that I do not use for imputation.

- Then, I use sklearn's `IterativeImputer` to impute the missing data.

- Note that I have abstracted the actual code away into functions, which are included in Section 2 of the notebook.

# Feature Engineering

**PS : Turned Out to be the Most Important work in this project, more essential than the modelling!!**

In this section I perform feature engineering and add many features to the data:
- I've abstracted all the code away into functions, please see Section 2 of this notebook for the actual code.

- In total, I engineer 28 new features, for which the names of each are included below.

- `median_facility_floor_site_eui`, `median_facility_year_site_eui`, and `median_facility_floor_year_site_eui` were the most powerful features for prediction.

- These features worked by concatenating a combination of the features `facility_type`, `floor_area`, and/or `year_built`, and then encoding them with a grouped median value of the target `site_eui`

- I thought that this would cause data leakage, since we are encoding the target variable as a feature (and in a sense, just identifying exact buildings in the dataset and using past `site_eui` to predict future `site_eui`, but in the end these three features were the most powerful in my entire notebook.

- This would suggest that a buildings prior site energy usage reading can be used as a key predictor in predicting it's future energy usage (similar to pure time series data).

Here are the features that were engineered from scratch.

<img src = "/blog/site-energy-intensity-pred/new_feat.PNG">

# Modelling

## Column Transformer

A few things to note on the final column transformer:

- You'll notice only a subset of features are used in the final model, and the majority of features are actually dropped.

- There were no categorical features used in my final solution.

- All features were numeric, and the only further preprocessing was to standardize them (not required for tree based learning, but can slightly improve performance).

Please refer my notebook for the code.

## Cross Validation

In this section, I perform cross validation.

- In the interest of speed, I used an out of the box light GBM regressor to perform five fold leave one group out cross validation ("LOGO cv").

- I performed LOGO cv based on the `year_factor` column in the dataset - in other words I trained and predicted on separate year groups, for example one fold would train on years 1 to 5, then predict on 6.

- I did this since the test dataset only included buildings from year 7, so I was trying to emulate this in cross validation by leaving one year out.  In a perfect world with more time, I would have also tried a fancier time series cross validation method.

- The mean cross validation scores for each grouped dataset are included below, we can see that for some groups, the train and validation scores are quite good, except for a few, namely `Food_Grocery`, `Health`, and `Laboratory_Data`.

<img src = "/blog/site-energy-intensity-pred/meancv.PNG">

## Feature Importance

- I plot the feature importance graphs from the lightGBM regressor models.

- I only include an example for the four `2to4_5plus_Mixed` models that resulted from the cross validation process, otherwise there would be too many plots to look at in this notebook.

- We'll see that in each fold the grouped median features and interaction features with `energy_star_rating` are the most important

<img src = "/blog/site-energy-intensity-pred/feat_imp.PNG">

## Final Models and Predictions

### Baseline RMSE calculation

```
avg_site_eui = [y_train.mean()] * len(y_train)
rmse = mean_squared_error(y_train, avg_site_eui, squared=False)
print('Baseline RMSE: %.3f' % (rmse))
```

The baseline RMSE I obtained was 0.57926. In the perfect world we cant get a RMSE of 0.0 but let's try to reduce it.

## Trying Out Different Models

### Linear Regression

```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train_lasso, y_train)
LinearRegression()
```
RMSE for Linear Regression I received is 0.47339

This is better than the baseline model! Let's explore more models :)

### Random Forest Model

```
rf = RandomForestRegressor()

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

cv_scores = cross_val_score(rf, X_train_lasso, y_train, 
                           scoring='neg_mean_squared_error', cv=cv, 
                           n_jobs=-1, error_score='raise', verbose=1)
```

**RMSE Score for Random Forest Model:**

```
cv_scores = np.absolute(cv_scores)
print('RMSE of random forest: %.3f' % (np.sqrt(cv_scores.mean())))
```
RMSE I received for Random Forest Model is 0.42317

### XGBoost Model

```
xgb = XGBRegressor(n_estimators=2000, learning_rate=0.2)
xgb.fit(X_train_lasso, y_train)

rmse = mean_squared_error(y_val, y_pred, squared=False)
print('RMSE of extreme gradient boosting: %.3f' % (rmse))
```
RMSE I received from XGBoost Model : 0.40337

<img src = "https://c.tenor.com/QE05ueMty3AAAAAM/thor-best-you-can-do.gif" height = 600px width = 600px>

### Does CatBoost give any increase in performance?

```
# code for CatBoost
from catboost import CatBoostRegressor

from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,eval_set=(X_val, y_val),plot=True)
```
Well the best RMSE I could get from CatBoost is 0.479, so we may have to look at ensembles to get the best out of the data!

### Ensemble of Boosting Models

- The final model I used was an ensemble of XGB, lightGBM, and CatBoost regressors.
- You will need to use a GPU to run this section of the notebook, or, comment out the GPU code lines (although this will take a long time if you only use a CPU).

PS. Running the model mlutiple times gives an error on the predictions after the 3rd decimal place, not sure why this happens even after random seed. I'll update the blog if I find out.

Please have a look at the [notebook](https://github.com/vedanthv/Site-Energy-Intensity-Prediction/blob/master/site-eui-pred-final.ipynb) for the code!

So I got the best RMSE score of 0.11 with the ensemble model. I did not notice any overfit[variance] or underfit[bias] whilst plotting the bias and variance curves but please DM on Twitter or LinkedIn [links in the home page] if you notice any overfit or underfit. You can also raise an [issue](https://github.com/vedanthv/Site-Energy-Intensity-Prediction/issues) on my GitHub repository.

# Possible Improvements and Better Performance🤔

## Kaggle Solutions Overview

### First Place Solution Overview

This was a great solution that included amazing feature engineering strategies and modelling. Read the discussions post by [jayjay](https://www.kaggle.com/jayjay75) [here](https://www.kaggle.com/competitions/widsdatathon2022/discussion/310522)

#### Summary of the Approach

* Main solution was based around finding the **previous history of the buildings**. Since the dataset did not have enough information about the past performance, features were engineered to calculate the past information. Bagging and regression turned out to be efficient in this case.

* Hundreds of other models were built with test set that didnt have previous information, LAG[don't worry I'll explain this in a bit!] features would not be useful in this case so pseudo labelling from the above point was used.

### How was feature engineering leveraged?

#### Weather Based Features 

* Statistical information like the mean,median, average and skew of temperatures was engineered.

* Season based binning was done and statistical parameters were extracted for each season.

* For the heating and cooling days, the statistical params / year was extracted and added as features.

* Here is the [link](https://www.kaggle.com/code/schopenhacker75/feature-engineering-catboost?scriptVersionId=89051266&cellId=14) to the weather based features code.

If you notice the code and the features I engineered, you can see that all these parameters were used to create new features.

But what did I not do? 

**Leverage LAG based features!**

#### LAG Based Features

**What are lag based features?** : a lagged variable has its value coming from an earlier point in time.

For example : **What was the Energy Usage Intensity of the building one, two or three years ago?**

The following lag based features were calculated in the solution.
* site_eui
* energy_star
* ELEVATION
* temp features

Here is the [code](https://www.kaggle.com/code/schopenhacker75/feature-engineering-catboost?scriptVersionId=89051266&cellId=19) for these features.

#### Rolling Based Features

Rolling Based Features: it consists in computing some statistical values (sum/mean..) for our target variables within a rolling window.

This method is called the rolling window method because the window would be different for every data point.

Example : **Moving average, over the last 1/2/3 years of the energy rating of the building**

Here is the [code](https://www.kaggle.com/code/schopenhacker75/feature-engineering-catboost?scriptVersionId=89051266&cellId=22) for the moving average calculation.

#### Delta Based features

Features extracted from the difference / variance / derivative of current value vs previous values were applied.

**For Example : we extracted the evolution rate of the Site Energy Comsumption from year-2 to year-1 then we constructed a new feature by multiplying this rate by the value of year-1.**

Here is the [code]("https://www.kaggle.com/code/schopenhacker75/feature-engineering-catboost?scriptVersionId=89051266&cellId=24") for the delta based features.

For some post-processing, pseudo labelling and training on models not dependent on previous information, click [here](https://www.kaggle.com/mathurinache/part2-wids2022-model2-no-lag)

# MLOps in Practice [Coming Soon!] 

This section has three main components.

* Building a Web Application with Streamlit

* Deploying the web application on Microsoft Azure

* Building Pipelines with Ploomber

* Model Monitoring with MLFlow

* Explanable AI and Model Interpretability with SHAP

