---
title: "🚦An End to End Project on Road Traffic Accident Severity Classification - Part 1/n"
description: "An end to end ML Solution to classify severity of road accidents using real world accident data"
dateString: July 2022
draft: false
tags: ["ML", "Python", "Projects"]
weight: 101
cover:
    image: "/blog/road-traffic-accident-classification/cover.jpeg"
---

# 📜 Problem Description
Every year the lives of approximately 1.3 million people are cut short as a result of a road traffic crash. Between 20 and 50 million more people suffer non-fatal injuries, with many incurring a disability as a result of their injury.

Road traffic injuries cause considerable economic losses to individuals, their families, and to nations as a whole. These losses arise from the cost of treatment as well as lost productivity for those killed or disabled by their injuries, and for family members who need to take time off work or school to care for the injured.

In this project, I'm going to describe an end to end process to classify road accident severity based on real world data.

# 📊 Source of the Dataset
This data set is collected from Addis Ababa Sub-city police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. 

All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms. 

Here is the [link]("https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591") to the dataset.

Now let's look at the in depth view of the pipelines, starting from preliminary analysis to model building and explanable AI.

# 📌 Pipelines and Indepth Analysis

## Phase 1 : Preliminary Data Analysis

The first thing to do in any data science project is to load the required libraries for the project. Here is a list of all the libraries that we are going to use :

* Numpy - Matrix Operations and Data Manipulation
* Pandas - Data Manipulation and Preprocessing
* Seaborn and Matplotlib - Data Visualization and Exploratory Data Analysis
* Plotly - Interactive 3D Plots
* Sklearn and XGBoost - Baseline Modelling and Hyperparameter Tuning
* Missingno - Analysis of Missing Data

Here is the code snippet to import all the libraries :
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (accuracy_score, 
                            classification_report,
                            recall_score, precision_score, f1_score,
                            confusion_matrix)

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

import shap

# this below code is just to ignore unnecessary warnings durning EDA
import warnings
warnings.filterwarnings('ignore')
```

**Loading the Dataset**
With all the libraries imported, let's load our dataset and start dabbling with the data!

```
# Since we have missing values represented as `na` and `NaN`, lets create an array for all the variations and pass it to the read_csv function.

missing_values = ['na','NaN']
df = pd.read_csv('dataset.csv',na_values = missing_values)
```

**How many rows and columns are there in the dataset?**

```
df.shape

# output : (12316, 32)
```

There are 12316 rows and 33 features in the dataset.

**What are the first 5 rows in the dataset?**
```
df.head()
```

```Output```

<img src = "/blog/road-traffic-accident-classification/data-5-rows.png" />

There are 32 features in the dataset, but 5 are presented here.

**List of all the features in the dataset**

```
df.columns
```

```Output```
<img src = "/blog/road-traffic-accident-classification/columns.png" />

**Now let's understand the description of some important features**

- time : time of the accident
- day_of_week : the day in which accident took place
- age_band_of_driver : the age bracket of the driver
- sex_of_driver : gender of the driver involved
- driving_experience : how experienced was the driver
- type of vehicle : what type of vehicle was involved
- owner of vehicle : who owned the vehicle?
- type_of_junction : what was the tpye of junction(Y-junction/T-Junction/U Turn/O-Junction)
- road_surface_type : what was the type of road(asphalt/gravel)
- accident_severity : what is the level of severity of the particular accident

We have an understanding of what each feature means in the dataset. Now let's understand the data types of all the features.

```
df.dtypes
```

On observing the output, we notice that only two of the columns `Number_of_vehicles_involved` and `Number_of_casualties` are numerical in nature whereas the others are of object type.

**Summary Statistics of Categorical Features**

By using the describe() function of pandas, we can generate the summary of the features.

```
df.describe()
```

```Output```
<img src = "/blog/road-traffic-accident-classification/describe.png">

**Observations :** 
On an average the *number of vehicles involved is 2* and the *number of causalities is 1*

The *maximum number of causalities is 7* and the *maximum number of vehicles involved is 8*

## Phase 2 : Data Preprocessing

**Are there any columns with missing values?**

```
df.isnull().any()
```

We can observe that most of he columns except `Time`,`Day_of_week`,`Age_band_of_driver`and `Sex_of_driver` has null values which needs to be processed.

**How many values are missing in each feature?**

```
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value_df = missing_value_df.reset_index()
```

Additionally, let's check which features heve the maximum missing values : 

```
missing_value_df.head()
```
<img src = "/blog/road-traffic-accident-classification/missing.png">

**Are there any outliers in the dataset?**

```
df.skew(axis=0)
```
**Output**
```
Number_of_vehicles_involved    1.323454
Number_of_casualties           2.344769
Casualty_severity             -2.893689
```
From the output we can observe that no conclusive measure of outliers can be derived from the dataset. 

**Dabbling with the target feature : ```Accident_severity```**

From this code snippet : ```df["Accident_severity"].unique()``` we can see that there are three main categories of our target variable : ```Slight Injury```,```Sever Injury``` and ```Fatal Injury```.

**Big Question : Is there a class imbalance in the target feature ?!**

The only way to find out is to plot a count plot.

```
ax = sns.countplot(y,label = 'Count')
Slight_Injury,Serious_Injury,Fatal = y.value_counts()
print("Slight Injury",Slight_Injury)
print("Serious Injury",Serious_Injury)
print("Fatal",Fatal)
```
**Output**
<img src = "/blog/road-traffic-accident-classification/imbalance.png" height = 300 width = 300>

We can see that there is class imbalance with over 10k values with `Slight_Injury` class and the other two classes have less than 15% values than the major class.

**Question  : What is the Mean Number of Vehicles Involed and Causalities Grouped By Accident_severity?**

```
df.groupby('Accident_severity').mean()
```
**Output** : 
<img src = "/blog/road-traffic-accident-classification/crosstab.png">

**Question : What is the association between cause_of_accident and Accident_severity?**

```
pd.crosstab(df["Accident_severity"],df['Cause_of_accident'])
```

**Output :**

<img src = "/blog/road-traffic-accident-classification/pivot.png">

**Key observations from the above graphs**

- Most Fatal injuries were caused by **Moving Backward**

- Most Serious injuries were caused by **Changing Lane to the right**

- Most Slight Injuries were caused by **No Distancing**

I'm going to cover the exploratory data analysis of the project in the upcoming post! Stay tuned!
