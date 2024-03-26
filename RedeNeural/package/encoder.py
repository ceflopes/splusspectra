from sklearn import preprocessing
import numpy as np
import pandas as pd
import os

from os import listdir
from os.path import isfile, join

encoders = {}

def encode_column(series : pd.Series, name : str):
    """Encode one column

    Args:
        series (pd.Series): pd.series or array to be encoded.
        name (str): name of column to be possible to save encoder.

    Returns:
        pd.Series : encoded columns 
    """    
    if name not in encoders:
        encoders[name] = preprocessing.LabelEncoder()
    res = encoders[name].fit_transform(series)
    return res


def decode_column(series : pd.Series, name : str):
    """Decode column

    Args:
        series (pd.Series): pd.series or array to be decoded.
        name (str): name of column, to find encoder.

    Returns:
        pd.Series : decoded column
    """    
    if name not in encoders:
        encoders[name] = preprocessing.LabelEncoder()
    res = encoders[name].inverse_transform(series)
    return res


##Decoding all char columns
def encode_DataFrame(df : pd.DataFrame):
    """Encodes all text dataFrame columns transforming them into numeric.

    Args:
        df (pd.DataFrame): DataFrame to transform. 

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """    

    str_columns = df.select_dtypes(include=[object]).columns
    for value in str_columns:
        try:
            df[value] = encode_column(df[value], value)
        except:
            pass
            
    return df

def save_encoders():
    """Save encoders fitted so it doesnt need to fit next time. 
    """    
    for name_encoder in encoders:
        np.save(os.path.join('package/encoders', name_encoder), encoders[name_encoder].classes_)


def load_encoders():
    """Load encoders saved. 
    """    
    onlyfiles = [f for f in listdir('package/encoders') if isfile(join('package/encoders', f))]
    for name_encoder in onlyfiles:
        encoders[name_encoder] = preprocessing.LabelEncoder()
        encoders[name_encoder].classes_ = np.load(os.path.join('package/encoders', name_encoder), allow_pickle=True)

    print("Loaded ", len(onlyfiles), "encoders.")\

load_encoders()