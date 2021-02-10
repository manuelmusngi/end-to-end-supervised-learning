#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# 
# 
# ## 1. Project Objective
# The objective of the end-to-end Supervised Learning project is to evaluate the effectivetiness of the 
# following models:
# 
# * Linear Regression
# * Decision Tree Regressor
# * Random Forest Regressor
# * Support Vector Regression
# 
# with hyperparameter optimization in the prediction of the median house values in California districts 
# and to consider the features in their predictive values.
# 
# 
# ## 2. Data  
# The data contains information from the 1990 California census from Kaggle Data set. Although the data set 
# is not up-to-date to predict current housing prices like the Zillow Zestimate data set, the data set however, 
# serves the basis for using Regression Evaluation in Supervised Learning. The data can be found on Kaggle: 
# 
# https://www.kaggle.com/camnugent/california-housing-prices
# 
# ## 3. Features for Evaluation
# The data features to the houses found in a given California district with summary statistics about them are 
# based on the 1990 census data. The columns or features are as follows:
# 
# longitude
# 
# latitude
# 
# housingmedianage
# 
# total_rooms
# 
# total_bedrooms
# 
# population
# 
# households
# 
# median_income
# 
# medianhousevalue
# 
# ocean_proximity
# 
# 
# ## 4. Evaluation Criteria
# The evaluation metric is the RMSE (root mean squared error). The root-mean-square deviation (RMSD) or the 
# root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or 
# population values) predicted by a model or an estimator and the values observed. In this case, the RMSE of 
# the median house prices.

# ## Import Libraries

# In[ ]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Main libraries
import numpy as np
import pandas as pd
import os
import urllib

# Visualization libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore warnings 
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Import data

# In[ ]:


HOUSING_PATH = '../input/california-housing-prices/housing.csv'
housing = pd.read_csv(HOUSING_PATH)


# ## Exploratory Data Analysis (EDA)

# In[ ]:


housing.head()


# In[ ]:


# features and their data types
housing.info()


# In[ ]:


# simple statistics about the data
housing.describe()


# In[ ]:


housing.describe().transpose()


# In[ ]:


housing["ocean_proximity"].value_counts()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()


# In[ ]:


# to make this notebook's output identical at every run
np.random.seed(42)


# In[ ]:


# extract training data and test data
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[ ]:


train_set.head()


# In[ ]:


test_set.head()


# In[ ]:


housing["median_income"].hist();


# In[ ]:


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[ ]:


housing["income_cat"].value_counts()


# In[ ]:


housing["income_cat"].hist();


# In[ ]:


# Provides train/test indices to split data in train/test sets
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[ ]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[ ]:


housing["income_cat"].value_counts() / len(housing)


# In[ ]:


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# In[ ]:


compare_props


# In[ ]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ## Data visualization

# In[ ]:


# Create a copy for visualization manipulation
housing = strat_train_set.copy()


# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")


# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")


# In[ ]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    # sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")


# In[ ]:


# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename));


# In[ ]:


import matplotlib.image as mpimg
california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()


# ### Correlation analysis

# In[ ]:


# Correlation of variables
corr_matrix = housing.corr()


# In[ ]:


mask = np.tril(housing.corr())
sns.heatmap(housing.corr(), fmt='.1g', annot=True, cmap='cool', mask=mask)


# In[ ]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[ ]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")


# In[ ]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")


# In[ ]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[ ]:


housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()


# ## Data preprocessing

# In[ ]:


housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


# In[ ]:


# find missing values
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[ ]:


# option 1
sample_incomplete_rows.dropna(subset=["total_bedrooms"]) 


# In[ ]:


# option 2
sample_incomplete_rows.drop("total_bedrooms", axis=1)       


# In[ ]:


# option 3
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) 


# In[ ]:


sample_incomplete_rows


# In[ ]:


# impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[ ]:


# Remove the text attribute because median can only be calculated on numerical attributes:
housing_num = housing.drop("ocean_proximity", axis=1)


# In[ ]:


imputer.fit(housing_num)


# In[ ]:


imputer.statistics_


# In[ ]:


# Manually compute the median of each attribute to check similarity with imputer.statistics_:
housing_num.median().values


# In[ ]:


# Transform the training set:
X = imputer.transform(housing_num)


# In[ ]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)


# In[ ]:


housing_tr.loc[sample_incomplete_rows.index.values]


# In[ ]:


imputer.strategy


# In[ ]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


# In[ ]:


housing_tr.head()


# In[ ]:


# preprocess the categorical input feature, ocean_proximity:
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[ ]:


ordinal_encoder.categories_


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array if 
# needed by calling the `toarray()` method:

# In[ ]:


# By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array if 
# needed by calling the `toarray()` method:
housing_cat_1hot.toarray()


# In[ ]:


# Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`:
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[ ]:


cat_encoder.categories_


# In[ ]:


# Create a custom transformer to add extra attributes: 
from sklearn.base import BaseEstimator, TransformerMixin

# Column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[ ]:


# housing_extra_attribs is a NumPy array, we've lost the column names (unfortunately, that's a problem 
# with Scikit-Learn).

# To recover a DataFrame, you could run this:
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# In[ ]:


# Build a pipeline for preprocessing the numerical attributes:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[ ]:


housing_num_tr


# In[ ]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[ ]:


housing_prepared


# In[ ]:


housing_prepared.shape


# In[ ]:


# Build a pipeline for preprocessing the numerical attributes:
from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[ ]:


# Join all these components into a big pipeline that will preprocess both the numerical and the 
# categorical features:
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])


# In[ ]:


from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])


# In[ ]:


old_housing_prepared = old_full_pipeline.fit_transform(housing)
old_housing_prepared


# The result is the same as with the `ColumnTransformer`:

# In[ ]:


np.allclose(housing_prepared, old_housing_prepared)


# # Modeling

# ### Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[ ]:


# Preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)


# In[ ]:


# Predictions
print("Predictions:", lin_reg.predict(some_data_prepared))


# In[ ]:


# Comparison against the actual values:
print("Labels:", list(some_labels))


# In[ ]:


some_data_prepared


# ####  Mean Squared Error metric

# In[ ]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# ####  Mean Absolute Error metric

# In[ ]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae


# ### Decision Tree Regressor Model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[ ]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[ ]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(scores)


# In[ ]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# ### Random Forest Regressor Model

# In[ ]:


# **Please Note**: we are specifying `n_estimators=100` since the default value is going to change to 
# 100 in Scikit-Learn 0.22.

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[ ]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[ ]:


# Cross validation for the RandomForestRegressor
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[ ]:


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


# ### Support Vector Regression Model

# In[ ]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# ## Hyperparameter Tuning

# ### GridSearch CV

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# ### Best hyperparameter combination suggested by GridSearchCV:

# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# ### Score of hyperparameter combination in grid search:

# In[ ]:


grid_search_cv_result = grid_search.cv_results_
for mean_score, params in zip(grid_search_cv_result["mean_test_score"], grid_search_cv_result["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:


pd.DataFrame(grid_search.cv_results_)


# ### Randomized Search CV

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, 
                                param_distributions=param_distribs,
                                n_iter=10, 
                                cv=5, 
                                scoring='neg_mean_squared_error', 
                                random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# In[ ]:


randomized_search_result = rnd_search.cv_results_
for mean_score, params in zip(randomized_search_result["mean_test_score"], randomized_search_result["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:


# Feature importance refers to techniques that assign a score to input features based on how useful they are 
# at predicting a target variable and the role of feature importance in a predictive modeling problem.

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[ ]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# ## Final suggested model

# In[ ]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[ ]:


print('The finalized evaluation of the Root Mean Squared Error (RMSE):', final_rmse)


# In[ ]:


# Compute a 95% confidence interval for the test RMSE:
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# In[ ]:


# Compute the interval:
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)


# In[ ]:


# Employing z-scores rather than t-scores: 
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# References: Aurelien Geron, Wikipedia 
