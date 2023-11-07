from sklearn import preprocessing
import numpy as np
import pandas as pd
import os
import joblib

from os import listdir
from os.path import isfile, join

encoders = {}


def decode_DataFrame(series : pd.Series, name : str):
    """Decode DataFrame

    Args:
        series (pd.Series): pd.series or array to be decoded.
        name (str): name of DataFrame, to find encoder.

    Returns:
        pd.Series : decoded column
    """    
    if name not in encoders:
        encoders[name] = preprocessing.MinMaxScaler(feature_range = (-1,1))
    res = encoders[name].inverse_transform(series)
    return res


##Decoding all char columns
def encode_DataFrame(df : pd.DataFrame, name : str):
    """Encodes all dataFrame columns transforming into range (-1, 1).

    Args:
        df (pd.DataFrame): DataFrame to transform. 
        name (string): DataFrame name.
    Returns:
        pd.DataFrame: Transformed DataFrame.
    """    
    if name not in encoders:
        encoders[name] = preprocessing.MinMaxScaler(feature_range = (-1,1))
    df = encoders[name].fit_transform(df)
            
    return df


def encode_Colors(df : pd.DataFrame, name : str):
    """Encodes columns individually transforming into range (-1, 1).

    Args:
        df (pd.DataFrame): DataFrame to transform.
        name (string): DataFrame name.
    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    nameC = name+"_colors"
    if (nameC) not in encoders:
        encoders[nameC] = preprocessing.MinMaxScaler(feature_range = (-1,1))
    for column in df.columns:
        df[column] = encoders[nameC].fit_transform(pd.DataFrame(df[column]))

    return df
    

def print_encoders():
    """Print all encoders saved. 
    """    
    for name_encoder in encoders:
        print(name_encoder)


def save_encoders():
    """Save encoders fitted so it doesnt need to fit next time. 
    """    
    for name_encoder, encoder in encoders.items():
        joblib.dump(encoder, os.path.join('package/encoders', name_encoder + '.pkl'))



def load_encoders():
    """Load encoders saved. 
    """    
    onlyfiles = [f for f in listdir('package/encoders') if isfile(join('package/encoders', f))]
    for name_encoder in onlyfiles:
        encoders[name_encoder.replace(".pkl", "")] = joblib.load(os.path.join('package/encoders', name_encoder))

    print("Loaded ", len(onlyfiles), "encoders.")

load_encoders()


def cap_outliers(df, column):
    
    upper = df[column].mean() + 3*df[column].std()
    down = df[column].mean() - 3*df[column].std()

    df[(df[column] > upper) | (df[column] < down)]

    df[column] = np.where(
        df[column]>upper,
        upper,
        np.where(
            df[column]<down,
            down,
            df[column]
        )
    )
    
    return df