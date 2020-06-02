"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
# import model

# Fetch training data and preprocess for modeling
# train = pd.read_csv('data/train_data.csv')
# riders = pd.read_csv('data/riders.csv')
# train = train.merge(riders, how='left', on='Rider Id')

# y_train = train[['Time from Pickup to Arrival']]
# X_train = train[['Pickup Lat','Pickup Long',
#                  'Destination Lat','Destination Long']]

# model._preprocess_data()
########### Rider dataset
riders = pd.read_csv('https://raw.githubusercontent.com/the-rick/regression_notebook_team17/master/data/Riders.csv')
# riders = pd.read_csv('data/riders.csv')

########### Train dataset
train_data = pd.read_csv('https://raw.githubusercontent.com/the-rick/regression_notebook_team17/master/data/Train.csv')
# train_data = pd.read_csv('data/train_data.csv')

######### merging datasets

#                                 TRAIN
df = pd.merge(train_data, riders,on = 'Rider Id',how='left')
         
########### Dropping vehicle type because it is always a bike

def drop_vehicle_type(input_df):
    input_df = input_df.drop(["Vehicle Type"], axis=1)
    return input_df
    
df = drop_vehicle_type(df)                   # TRAIN DATA

########### Assigning features and predictor variables

#                               TRAIN DATA
X = df.drop(["Time from Pickup to Arrival"],axis=1)
y = df.iloc[:,-1].values

"""
Dropping these features because the test data does not contain them
"""

X = X.drop(['Arrival at Destination - Day of Month','Arrival at Destination - Weekday (Mo = 1)', 'Arrival at Destination - Time'], axis=1)

########### Dealing with missing values

"""
PREDICTION VARIABLE
Dropping missing values for Time from Pickup to Arrival
"""
y = y[~np.isnan(y)]

"""
PRECIPITATION
Missing values will be replaced with 0. The zero will mean that there was no precipitation 
during that day.
"""

X["Precipitation in millimeters"] = X["Precipitation in millimeters"].fillna(0)  #TRAIN DATA

"""
TEMPERATURE
Filling NaN values with the mean of the column
"""
def impute_mean(series):
    return series.fillna(series.mean())
    
X["Temperature"] = round(X.Temperature.transform(impute_mean),1)     # TRAIN DATA

"""
Drop the riders from the Rider dataset who do not have information on the train data and test data
Number of rows will go from 21237 to 21201 in train dataset
Number of rows will go from 7206 to 7068 in test dataset
"""

def drop_nan_rows(input_df):
    input_df = input_df.dropna(how='any', subset=['User Id'])
    return input_df
    
X = drop_nan_rows(X)                    # TRAIN DATA

############ Categorising data and encoding it

"""
RIDER ID
   
Creating a count variable that counts the number of times each rider ID appeArs,
then breaking the counts values into categorical values to reduce the number of dummy variables
"""


                                                  
                    # TRAIN DATA
X["Counts"] = X.groupby("Rider Id")["Order No"].transform('count')
X["Is_rider_busy"] = X["Counts"].apply(lambda x: 1 if x >= 50 else 0)

"""
Dropping the counts and Rider Id columns, after utilizing them
"""

X = X.drop(["Rider Id","Counts"],axis = 1)              # TRAIN DATA

"""
USER  ID
  
Doing the same method we did for the rider ID variable to reduce the number of dummy variables
A user will be a frequent user if they are returning for the second time or more,
if a user appears once on the list, they are regarded as non-frequent
"""

def user_id_cat(input_df):
    input_df["Counts"] = input_df.groupby("User Id")["Order No"].transform('count')

    input_df["Is_user_frequent"] = input_df["Counts"].apply(lambda x: "Frequent" if x >= 135 
                                else "Moderate" if 10 < x < 135 
                                else  "Occasional" )
    return input_df

X = user_id_cat(X)

"""
Dropping the counts and User Id columns, after utilizing them
"""
X = X.drop(["User Id","Counts"],axis = 1)               # TRAIN DATA

"""
PLATFORM TYPE
  
Because on platform type 3 was used for approximately 85.16% of the orders placed in the dataset, it will be regarded as the
busiest platform, and the others will be regarded as not busy.
"""

def platfor_type(input_df):
    input_df["Is_platform_busy"] = input_df["Platform Type"].apply(lambda x: 1 if x == 3 
                                    else 0)
    return input_df
    
X = platfor_type(X)                   # TRAIN DATA

"""
Dropping Platform Type column after utilizing it
"""
X = X.drop(["Platform Type"],axis = 1)               # TRAIN DATA

"""
ORDER NO
   
Dropping Order No column because  it is unique for every row
"""
X = X.drop(["Order No"],axis=1)                        # TRAIN DATA

"""
Making sure column naming is consistent by replacing whitespaces with an underscore
"""

X.columns = [col.replace(" ","_") for col in X.columns]                   # TRAIN DATA

########### Encoding features

objList = X.select_dtypes(include = "object").columns            #finding features with type object
objList
excluding_time = ['Personal_or_Business','Is_user_frequent']

X = pd.get_dummies(X, columns=excluding_time,drop_first = True)
    
########## dropping just to test the models
X = X.drop(['Placement_-_Time','Confirmation_-_Time','Arrival_at_Pickup_-_Time',
                       'Pickup_-_Time'],axis = 1)
    
import statsmodels.api as sm

#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols

"""
Dropping features because they are highly correlated
"""

selected_features_BE.remove('No_of_Ratings')
selected_features_BE.remove('Is_user_frequent_Moderate')

X_copy_scaling = X[selected_features_BE]

sc = StandardScaler()
X_copy_scaling.iloc[:,0:6]= sc.fit_transform(X_copy_scaling.iloc[:,0:6])

X_copy_split = X_copy_scaling.copy()
X_array = np.array(X_copy_split)

kf = KFold(n_splits=4) # Define the split - into 2 folds 
kf.get_n_splits(X_array)

train_tuples = []
test_tuples = []

for train_index, test_index in kf.split(X_array ):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train_tuples.append((X_train,y_train))
    test_tuples.append((X_test,y_test))

# Testing model using test data taken from train data
model_kfold = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=9, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)

for (X_train,y_train) in train_tuples:
    model_kfold.fit(X_train,y_train)  #training model


# Fit model
# lm_regression = LinearRegression(normalize=True)
# print ("Training Model...")
# lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/team_17_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(model_kfold, open(save_path,'wb'))
