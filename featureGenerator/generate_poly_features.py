"""
This script is part of a transition towards Python 3.x support.

What it does?
    - Remove unused features
    - Creates polynomial features

Issues?
    - Only for Rutgers dataset

"""
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures 

from scipy.io import arff


def generate_polynomial_features(data, exclude=[], *args, **kwargs):
    """
    data: pd.DataFrame
    exclude: List[str]
    *args, **kwargs are passed to sklearn.preprocessing.PolynomialFeatures
    """
    X, y = data.drop(exclude, axis=1), data[exclude]
    poly = PolynomialFeatures(*args, **kwargs).fit(X)

    # Next line converts back to pandas, while keeping column names
    X = pd.DataFrame(poly.transform(X), columns=poly.get_feature_names(X.columns))
    
    return pd.concat([X, y], axis=1)


def rename_columns(column_name: str) -> str:
    return column_name.replace(' ', '*')


if __name__ == '__main__':
    data = arff.loadarff('./output/output.arff')

    df = pd.DataFrame(data[0])

    # Show initial columns
    print('Input:')
    df.info()


    # Drop unused features. It will simplify further process
    df.drop(['seq', 'link_num', 'experiment_num', 'dataset_num', 'received', 'prr'], axis=1, inplace=True)
    df['class'] = df['class'].apply(lambda v: v.decode('ascii')) # For some reason output of `generate_features.py` is b'<feature_name>'

    # Show how it goes
    print('After dropping unused features:')
    df.info()


    # Polynomial features
    df = generate_polynomial_features(df, exclude=['class'], degree=2, include_bias=False)
    # Replace space with asterisk for multiplication
    df = df.rename(rename_columns, axis='columns')

    # Output to CSV file
    df.to_csv('./output/output_stage3.csv', index=False)

    print('Final dataset description:')
    df.info()
