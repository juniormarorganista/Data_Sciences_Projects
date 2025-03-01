# -*- coding: utf-8 -*-
"""Pandas Course HOMEWORK Solved.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10nfTnaVxsoCmwWfTnVcX8joa4un37-UT

# Pandas Course HOMEWORK - solved
"""

!pip install pandas
#!pip install pandas==2.0.3

import pandas as pd
import numpy as np
print(pd.__version__)
print(np.__version__)

"""# Series

- Dataset: https://www.kaggle.com/datasets/rakannimer/air-passengers
"""

dataset = pd.read_csv('/content/AirPassengers.csv')

type(dataset)

dataset

serie = pd.Series(np.array(dataset['#Passengers']), index = dataset['Month'])
serie

type(serie)

serie.index

serie.dtype

# Number of dimensions
serie.ndim

serie.size

serie.head()

serie.tail()

serie.iloc[0]

serie.iloc[0:4]

serie.iloc[-1]

serie.loc["1960-08"]

serie.loc["1960-08":"1960-12"]

# Ordering
serie.sort_values()

# Ordering
serie.sort_values(ascending = False)

# Ordering
serie.sort_index()

# Counting
serie.value_counts()

# Greater than 300
serie.loc[serie > 300]

serie.loc[serie.index == "1954-07"]

# Less than 1950-08
serie.loc[serie.index < "1950-08"]

# Between 1949-01 and 1950-01
serie.loc[(serie.index >= "1949-01") & (serie.index < "1950-01")]

# Sum
serie.sum()

# Average
serie.mean()

# Minimum value
serie.min()

# Maximum value
serie.max()

# Sum between 1949-01 and 1949-12
serie.loc[(serie.index >= "1949-01") & (serie.index < "1950-01")].sum()

# Unique values
serie.index.unique()

# Sum of not null values
serie.isna().sum()

serie

# Search for pd.DatetimeIndex to extract the month from the index
pd.DatetimeIndex(serie.index).month

# Search for pd.DatetimeIndex to extract the year from the index
pd.DatetimeIndex(serie.index).year

# Sum of month 7
serie.loc[(pd.DatetimeIndex(serie.index).month == 7)].sum()

# Sum of year 1950
serie.loc[(pd.DatetimeIndex(serie.index).year == 1950)].sum()

"""# Dataframe

- Read the games.csv dataset, showing the basic information
"""

dataset = pd.read_csv('games.csv')

dataset.info()

dataset

"""- Basic statistics"""

dataset.describe()

dataset.shape

dataset.columns

"""- Sum of missing values"""

dataset.isna().sum()

"""- Delete all missing values rows"""

dataset = dataset.dropna()

dataset.shape

dataset

"""- Check if duplicated rows exist"""

dataset.duplicated().sum()

dataset.dtypes

"""- Convert the data types, according to the types listed below"""

dataset = dataset.astype({"Year_of_Release": "int", "Critic_Count": "int", "User_Count": "int", "User_Score": "float64"})

dataset.dtypes

dataset.head()

"""- Create a function to update the User_Score attribute, to make it on the same scale as the Critic_Score attribute (just multiply by 10)
- Use apply to update the dataframe
"""

def update_critic(score):
  return score * 10

dataset["User_Score"] = dataset["User_Score"].apply(update_critic)

dataset.head()

dataset.iloc[0:5,0:5]

"""- Delete the following columns ["NA_Sales", "JP_Sales", "EU_Sales", "Other_Sales", "Critic_Count", "User_Count"]"""

dataset.drop(["NA_Sales", "JP_Sales", "EU_Sales", "Other_Sales", "Critic_Count", "User_Count"], axis = 1, inplace = True)

dataset.head()

"""- Show count of values for each categorical column"""

for column in dataset.columns:
  if dataset[column].dtype == object:
    print('------- ', column, '---------')
    print(dataset[column].value_counts(normalize = True))
    print()

"""- The 5 best-selling games"""

dataset.sort_values(["Global_Sales"], ascending = [False]).head(5)

"""- The 5 least sold games"""

dataset.sort_values(["Global_Sales"], ascending = [True]).head(5)

"""- All PS4 games, showing only the following columns: name, year_of_release e developer"""

dataset.loc[dataset["Platform"] == "PS4", ["Name", "Year_of_Release", "Developer"]]

"""- PS4 games with User_Score greater than 90"""

dataset.loc[(dataset["Platform"] == "PS4") & (dataset["User_Score"] > 90)]

"""- PS4 games with User_Score greater than 85 (use the query function)"""

dataset.query("Platform == 'PS4' and `User_Score` > 85")

"""- Create a new column, which is the average of Critic_Score and User_Score"""

dataset["Score"] = (dataset["Critic_Score"] + dataset["User_Score"]) / 2

"""- Delete Critic_Score and User_Score columns"""

dataset.drop(["Critic_Score", "User_Score"], axis = 1, inplace = True)

dataset

dataset.columns

"""- Reposition the columns in the order below"""

dataset = dataset.reindex(labels = ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher',
       'Global_Sales', 'Score', 'Developer', 'Rating'], axis = 1)

dataset

dataset.info()

"""- Change the type of all category columns to type "category"
"""

dataset = dataset.astype({"Name": "category", "Platform": "category", "Genre": "category",
                          "Publisher": "category", "Developer": "category", "Rating": "category"})

"""- Check the amount of memory used"""

dataset.info()

"""- The sum of Global_Sales"""

dataset.loc[:, ["Global_Sales"]].sum()

"""- The median of Global Sales"""

dataset.loc[:, ["Global_Sales"]].mean()

"""- The unique values of Platform"""

dataset["Platform"].unique()

"""- Platform Grouping by Global Sales"""

dataset.groupby("Platform")["Global_Sales"].mean().sort_values(ascending = False)

"""- Genre and Platform grouping, showing the average of other attributes (agg function)"""

dataset.groupby(["Genre", "Platform"]).agg("mean", numeric_only = True)

"""- Sales count by genre"""

dataset.groupby("Genre")["Global_Sales"].count().sort_values(ascending = False)

"""- The average Global Sales by gender"""

group = dataset.groupby(["Genre"], as_index=False)["Global_Sales"].mean().sort_index()
group

"""- Add a new column with the average Score (transform function)"""

group.assign(avg_score = dataset.groupby(["Genre"])["Score"].transform("mean"))

"""- Create a pivot table to represent the data as shown in the table below"""

dataset.pivot_table(index = "Genre",
                    columns = "Platform",
                    values = "Global_Sales",
                    aggfunc = "mean")

"""# Plots

- Load the games.csv base and delete the missing values
"""

dataset = pd.read_csv('games.csv')
dataset = dataset.dropna()

dataset.head()

"""- Visualizing subgraphs of numeric attributes"""

dataset.plot(subplots = True, layout = (3,3), figsize = (11,11),
             legend = False, kind = "hist");

"""- Count of unique values of the Platform attribute"""

dataset["Platform"].unique()

"""- Number of games per platform"""

dataset.groupby("Platform")["Platform"].count().plot(kind = "bar");

"""- Game count pie chart by platform"""

dataset.groupby("Platform")["Platform"].count().plot.pie(title = "Platform", ylabel = "");

"""- Viewing dataset columns"""

dataset.columns

"""- Scatterplot between Critic_Score and Global_Sales"""

dataset.plot.scatter(x = "Critic_Score", y = "Global_Sales");

"""- Critic_Score column histogram"""

dataset["Critic_Score"].plot.hist();