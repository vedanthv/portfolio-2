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
- You can see the exact groups of facility types that I used by reading the `get_manual_facility_groups` function in the Helper Sections [2nd Section] of the Notebook 