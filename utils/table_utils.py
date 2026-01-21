import os
import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def EncodeFeatures(df, use_features, dictPath):
    result_df = df.copy()
    category_feat = []
    
    for feat in use_features:

        if feat == 'uniprot_id':
            map_dict = json.load(open(os.path.join(ROOTDIR, dictPath)))
            result_df[feat] = result_df[feat].map(map_dict)
            category_feat.append(feat)

            if result_df[feat].isnull().any():
                result_df[feat] = result_df[feat].fillna(-1)

        elif isinstance(df[feat].values[0], str):
            category_feat.append(feat)
            pass

        else:
            result_df[feat] = result_df[feat].astype('float32')

    return result_df, category_feat

def CDSContext(df):
    pass