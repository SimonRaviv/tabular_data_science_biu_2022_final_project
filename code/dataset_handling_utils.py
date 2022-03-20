"""
@brief This module supports the file reading and datasets preprocessing.

Read the preprocess and 4 required datasets .
"""
import pandas as pd

# Dataset utility functions
def preprocess_houses(df):
    df.fillna(df.mean(), inplace=True)
    del df['Id']
    return df

def preprocess_cancer(df):
    del df['id']
    del df['Unnamed: 32']
    return df

def preprocess_sleeping(df):
    return df

def preprocess_salaries(df):
    del df['Id']
    del df['Notes']
    df.Benefits.fillna(0, inplace=True)
    df = df[pd.to_numeric(df.Benefits,errors='coerce').notna()]
    df[df.BasePay.isna()] = 0
    for c in ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']:
        df[c] = df[c].astype('float64')
    return df
    
def get_numeric_df(df):
    numeric_columns = df.dtypes[(df.dtypes == "float64") | (df.dtypes == "int64")].index.tolist()
    very_numerical = [nc for nc in numeric_columns if df[nc].nunique() > 20]
    return df[very_numerical]

def read_dataset(file_path, preprocess_func, only_numeric_data=False):
    dtf = pd.read_csv(file_path, low_memory=False)
    dtf = preprocess_func(dtf)
    if only_numeric_data is True:
        return get_numeric_df(dtf)
    return dtf

# Datasets metadata
class DS:
    """
    @brief: Datasets names.
    """
    HOUSES = "houses"
    CANCER = "cancer"
    SLEEPING = "sleeping"
    SALARIES = "salaries"
