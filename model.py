"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
import statsmodels.api as sm

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df
    
    ########### Rider dataset
    riders = pd.read_csv('https://raw.githubusercontent.com/the-rick/regression_notebook_team17/master/data/Riders.csv')
    
    ########### Train dataset
    train_data = pd.read_csv('https://raw.githubusercontent.com/the-rick/regression_notebook_team17/master/data/Train.csv')
    
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

    predict_vector = predict_vector[selected_features_BE]

    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
