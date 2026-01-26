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

        elif feat == 'mut+res-bin':
            result_df[feat] = result_df[feat].astype('int32')
            category_feat.append(feat)

        elif feat == 'mutant_residue_types_list':
            result_df[feat] = result_df[feat].astype('str')
            category_feat.append(feat)

        else:
            result_df[feat] = result_df[feat].astype('float32')

    return result_df, category_feat

def CDSContext(df):
    pass

def make_bin_cols(
    df: pd.DataFrame, 
    gen_col_name: str, 
    bin_size: int = 42, 
    method: str = 'bin'
) -> pd.DataFrame:
    """
    Bins mutability data and plots the distribution.
    method: 'bin' (equal width) or 'q' (quantile based)
    """
    if method == 'bin':
        df[gen_col_name] = pd.cut(df['mutability'], bins=bin_size, labels=False, include_lowest=True)
    elif method == 'q':
        df[gen_col_name] = pd.qcut(df['mutability'], q=bin_size, labels=False, duplicates='drop')

    df[gen_col_name] = df[gen_col_name].astype('int32')

    return df


def scaling_and_fillnafeature(df, feat_name=None):

    if feat_name is None:
        return df

    else:
        result_df = df.copy()

        for feat in feat_name:
            if "DAYM780301" in feat:
                result_df[feat] = 10**(result_df[feat]/10)
                result_df[feat].fillna(0, inplace=True)
                print(f"Scaling {feat}:", "10^(DAYM/10) & Fillna(0)")
            elif "HENS920102" in feat:
                result_df[feat] = 2**(result_df[feat]/3)
                result_df[feat].fillna(0, inplace=True)
                print(f"Scaling {feat}:", "2^(HENS/3) & Fillna(0)")

            elif 'unique_patients_count' in feat:
                result_df[feat].fillna(0, inplace=True)
                print(f"Scaling {feat}:", "Fillna(0)")
            elif 'total_mutations_count' in feat:
                result_df[feat].fillna(0, inplace=True)
                print(f"Scaling {feat}:", "Fillna(0)")
            elif 'unique_mutation_types_count' in feat:
                result_df[feat].fillna(0, inplace=True)
                print(f"Scaling {feat}:", "Fillna(0)")

            else:
                raise ValueError(f"Feature {feat} is not supported.")
                
    return result_df