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

## Overview and Abstract 

Climate change is a globally relevant, urgent, and multi-faceted issue heavily impacted by energy policy and infrastructure. Addressing climate change involves mitigation (i.e. mitigating greenhouse gas emissions) and adaptation (i.e. preparing for unavoidable consequences). Mitigation of GHG emissions requires changes to electricity systems, transportation, buildings, industry, and land use.

the lifecycle of buildings from construction to demolition were responsible for 37% of global energy-related and process-related CO2 emissions in 2020. Yet it is possible to drastically reduce the energy consumption of buildings by a combination of easy-to-implement fixes and state-of-the-art strategies.

## Overview : Dataset

The WiDS Datathon dataset was created in collaboration with Climate Change AI (CCAI) and Lawrence Berkeley National Laboratory (Berkeley Lab). WiDS Datathon participants will analyze differences in building energy efficiency, creating models to predict building energy consumption. Participants will use a dataset consisting of variables that describe building characteristics and climate and weather variables for the regions in which the buildings are located. Accurate predictions of energy consumption can help policymakers target retrofitting efforts to maximize emissions reductions.

## Brief Description of the Solution 

> The project involves predicting a buildings Site Energy Usage Intensity metric, which was the `site_eui` feature in the dataset (regression problem).  To solve this problem, I first separated the dataset into twelve individual datasets based on buildings with similar `site_eui` usage patterns and other characteristics.  I then engineer features, perform leave one group out cross validation, and finally train an ensemble model (XGBoost, LightGBM, and CatBoost regressors) for each dataset on the most powerful features.  My final solution ended up in the top 10% of the final leaderboard.

## Helper Functions

To make the code in the projects modular, I have used helper functions that include reusable code for **reading data**, **preprocessing tasks**, **feature engineering**, **cross validation** and **modelling**. Feel free to check out my [github](https://github.com/vedanthv/Site-Energy-Intensity-Prediction/blob/master/site-eui-pred-final.ipynb) repository for indepth docstrings of each function and post an [issue](https://github.com/vedanthv/Site-Energy-Intensity-Prediction/issues) if you want to report a bug for any of the functions.

## Data Preprocessing

### Dealing with Duplicate Data

* We see that there are 39 duplicated buildings in the train set and 5 duplicate buildings in the test set. 

* Since the number of duplicates is very less compared to the 75k samples in the dataset, I have removed the duplicates.

* I have left the duplicates in the test set as otherwise, it could affect the predictions.

## Feature Engineering

### Create Individual Datasets Based on Facility Types

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

## Missing Data

In this section I identify what data is missing in the dataset:

- We'll see that features related to fog or wind, have over 50% of the data missing in both the train and test set, and due to this, I don't use these features in my final solution.

- We'll also see that there is missing data in energy star rating and year built, which I deal with in the next section of this blog.

In this section, I impute the missing data in the `energy_star_rating` and `year_built` features:
- I use Ridge regression to impute the missing values, as I found this gave me the best results in terms of optimizing the final score of my solution.

- I also tried using XGBoost and LightGBM models to impute, but these did not do as well.
- First, I create a sklearn `ColumnTransformer` which one hot encodes the categorical features I used for imputation, as well as removes features that I do not use for imputation.

- Then, I use sklearn's `IterativeImputer` to impute the missing data.

- Note that I have abstracted the actual code away into functions, which are included in Section 2 of the notebook.

## Feature Engineering

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

## Modelling

### Column Transformer

A few things to note on the final column transformer:

- You'll notice only a subset of features are used in the final model, and the majority of features are actually dropped.

- There were no categorical features used in my final solution.

- All features were numeric, and the only further preprocessing was to standardize them (not required for tree based learning, but can slightly improve performance).

Please refer my notebook for the code.

### Cross Validation

In this section, I perform cross validation.

- In the interest of speed, I used an out of the box light GBM regressor to perform five fold leave one group out cross validation ("LOGO cv").

- I performed LOGO cv based on the `year_factor` column in the dataset - in other words I trained and predicted on separate year groups, for example one fold would train on years 1 to 5, then predict on 6.

- I did this since the test dataset only included buildings from year 7, so I was trying to emulate this in cross validation by leaving one year out.  In a perfect world with more time, I would have also tried a fancier time series cross validation method.

- The mean cross validation scores for each grouped dataset are included below, we can see that for some groups, the train and validation scores are quite good, except for a few, namely `Food_Grocery`, `Health`, and `Laboratory_Data`.




