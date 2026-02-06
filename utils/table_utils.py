import os
import ast
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


# ==========================================
# Feature Configuration
# ==========================================

# 1. External Mapping Configuration
MAPPING_CONFIG = {
    'uniprot_id': {'filename': 'uniprot-id_to_index.json', 'fillna': 0},
    'mut+res-bin': {'filename': 'mut+res-bin_mapping.json', 'fillna': 0},
    'dssp_sec_struct': {'filename': 'dssp_sec_struct_mapping.json', 'fillna': 0},
    'dssp_helix_3_10': {'filename': 'dssp_helix_3_10_mapping.json', 'fillna': 0},
    'dssp_helix_alpha': {'filename': 'dssp_helix_alpha_mapping.json', 'fillna': 0},
    'dssp_helix_pi': {'filename': 'dssp_helix_pi_mapping.json', 'fillna': 0},
    'dssp_helix_pp': {'filename': 'dssp_helix_pp_mapping.json', 'fillna': 0},
    'dssp_sheet': {'filename': 'dssp_sheet_mapping.json', 'fillna': 0},
    'dssp_strand': {'filename': 'dssp_strand_mapping.json', 'fillna': 0},
    'dssp_ladder_1': {'filename': 'dssp_ladder_1_mapping.json', 'fillna': 0},
    'dssp_ladder_2': {'filename': 'dssp_ladder_2_mapping.json', 'fillna': 0},
}

# 2. Angular Features (Sin/Cos Transform)
ANGULAR_FEATURES = {'dssp_alpha', 'dssp_phi', 'dssp_psi'}

# 3. Numeric Features with NaN Indicators 
# Key: Feature Name, Value: Fillna Value
NUMERIC_NA_CONFIG = {
    'dssp_accessibility': -1,
    'dssp_TCO': -2,
    'dssp_kappa': -1,
}

def EncodeFeatures(df, use_features, verbose=True):
    result_df = df.copy()
    category_feat = []
    numerical_feat = []

    for feat in use_features: # from yaml files
        # 1. Handle Mapping File Features
        if feat in MAPPING_CONFIG:
            config = MAPPING_CONFIG[feat]
            if feat == 'uniprot_id':
                 path = os.path.join(ROOTDIR, 'config/mapping/uniprot-id_to_index.json')
            else:
                 path = os.path.join(ROOTDIR, 'config/mapping', config['filename'])
            
            result_df = apply_and_save_mapping(result_df, [feat], path)
            if config['fillna'] is not None:
                result_df[feat].fillna(config['fillna'], inplace=True)
            
            category_feat.append(feat)
            
        # 2. Handle Angular Features (Sin/Cos)
        elif feat in ANGULAR_FEATURES:
            result_df, new_cols = process_angular_feature(result_df, feat, verbose)
            numerical_feat.extend(new_cols)

        # 3. Handle Numeric Features with NaN Indicators
        elif feat in NUMERIC_NA_CONFIG:
            fill_val = NUMERIC_NA_CONFIG[feat]
            result_df, new_cols = process_numeric_na_feature(result_df, feat, fill_val, verbose)
            numerical_feat.extend(new_cols)

        # 4. Handle Specific Custom Features
        elif feat == 'dssp_bend':
            result_df[feat] = result_df[feat].map({'S': 1})
            result_df[feat].fillna(0, inplace=True)
            numerical_feat.append(feat)
            if verbose:
                print("dssp_bend: fillna(0) and map S to 1")

        elif feat == 'dssp_chirality':
            result_df[feat] = result_df[feat].map({'-': 1, '+': 2})
            result_df[feat].fillna(0, inplace=True)
            numerical_feat.append(feat)
            if verbose:
                print("dssp_chirality: fillna(0) and map -/+ to 1/2")

        elif feat == 'ptms_mapped':
            result_df, new_cols = process_ptms(result_df, verbose)
            numerical_feat.extend(new_cols)

        # 3. Handle Error for Unknown Features
        # else:
        #     raise ValueError(f"Fail to encode feature {feat} || Check table_utils.py")

    return result_df, category_feat, numerical_feat

def process_angular_feature(df, feat, verbose=True):
    """Handles Sin/Cos transformation and NaN indicator."""
    col_na = f'{feat.replace("dssp_", "")}_is_na' # e.g. alpha_is_na
    df[col_na] = df[feat].isna().astype('int8')
    
    rad = np.radians(df[feat])
    df[f'{feat}_sin'] = np.sin(rad).astype('float32')
    df[f'{feat}_cos'] = np.cos(rad).astype('float32')
    
    df[f'{feat}_sin'].fillna(0, inplace=True)
    df[f'{feat}_cos'].fillna(0, inplace=True)
    if verbose:
        print(f"Processing {feat}: Sin/Cos conversion & {col_na} created")
    return df, [col_na]


def process_numeric_na_feature(df, feat, fill_val, verbose=True):
    """Handles numeric features with NaN indicator and fill value."""
    # Suffix for indicator: usually '_is_na' but original code had specifics like 'tco_is_na'
    # We'll use split logic to match original naming convention if possible or standardize.
    # Original: dssp_accessibility -> accessibility_is_na
    # Original: dssp_TCO -> tco_is_na
    suffix = feat.replace("dssp_", "").lower() + "_is_na"
    if 'TCO' in feat: suffix = 'tco_is_na' # special casing to match exact original output if strict

    df[suffix] = df[feat].isna().astype('int8')
    df[feat].fillna(fill_val, inplace=True)
    if verbose:
        print(f"Processing {feat}: fillna({fill_val}) & {suffix}")
    return df, [suffix]


def process_ptms(df, verbose):
    """Handles one-hot encoding of PTM lists."""
    df['ptms_mapped'] = df['ptms_mapped'].apply(parse_ptm)
    all_ptms = set()
    for l in df['ptms_mapped']:
        all_ptms.update(l)
    all_ptms = sorted(list(all_ptms))

    new_cols = []
    for ptm in all_ptms:
        col_name = f'ptm_{ptm}'
        df[col_name] = df['ptms_mapped'].apply(lambda x: 1 if ptm in x else 0).astype('int8')
        new_cols.append(col_name)
        
    df['ptm_is_na'] = (df['ptms_mapped'].apply(len) == 0).astype('int8')
    new_cols.append('ptm_is_na')
    
    df.drop(columns=['ptms_mapped'], inplace=True)
    if verbose:
        print(f"Processing ptms_mapped: One-hot encoding & ptm_is_na created")
    return df, new_cols


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

    df[gen_col_name] = df[gen_col_name].astype('int8')

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
                result_df['mut_is_na'] = result_df[feat].isna().astype('int8')
                result_df[feat].fillna(0, inplace=True)
                print(f"Scaling {feat}:", "Fillna(0)")
            elif 'unique_mutation_types_count' in feat:
                result_df[feat].fillna(0, inplace=True)
                print(f"Scaling {feat}:", "Fillna(0)")                
    return result_df


def apply_and_save_mapping(df, columns, mapping_path):
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            full_mapping = json.load(f)

    else:
        full_mapping = {}

    for col in columns:
        # Fill NaN with 'Unknown' first
        df[col] = df[col].fillna('Unknown').astype(str)
        
        if col not in full_mapping:
            unique_labels = sorted(df[col].unique())
            # Force 'Unknown' to be 0
            if 'Unknown' in unique_labels:
                unique_labels.remove('Unknown')
            
            # Start indexing from 1 for valid labels
            mapping = {label: i for i, label in enumerate(unique_labels, 1)}
            mapping['Unknown'] = 0 # Explicitly set Unknown to 0
            
            full_mapping[col] = mapping

        # Map values. default to 0 (Unknown) if not found
        unknown_idx = full_mapping[col].get('Unknown', 0)
        df[col] = df[col].map(lambda x: full_mapping[col].get(x, unknown_idx))

    if not os.path.exists(mapping_path):
        with open(mapping_path, 'w') as f:
            json.dump(full_mapping, f, indent=4)
    
    return df


def parse_ptm(x):
    if pd.isna(x) or x == '[]':
        return []
        # "['ac', 'for']" -> ['ac', 'for']
    return ast.literal_eval(x)