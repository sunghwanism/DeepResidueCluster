
import os
import ast
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================
# Feature Configuration
# ==========================================

MAPPING_CONFIG: Dict[str, Dict[str, Any]] = {
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

ANGULAR_FEATURES: Set[str] = {'dssp_alpha', 'dssp_phi', 'dssp_psi'}

NUMERIC_NA_CONFIG: Dict[str, int] = {
    'dssp_accessibility': -1,
    'dssp_TCO': -2,
    'dssp_kappa': -1,
}

def encode_features(df: pd.DataFrame, use_features: List[str], mapping_dir: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Encodes features in the DataFrame based on configuration.
    
    Args:
        df: Input DataFrame.
        use_features: List of features to use.
        mapping_dir: Directory containing mapping JSON files.
        
    Returns:
        Tuple of (Processed DataFrame, Categorical Feature Names, Numerical Feature Names)
    """
    result_df = df.copy()
    category_feat: List[str] = []
    numerical_feat: List[str] = []

    for feat in use_features:
        # 1. Handle Mapping File Features
        if feat in MAPPING_CONFIG:
            config = MAPPING_CONFIG[feat]
            # Handle specific path for uniprot_id if originally hardcoded, 
            # but ideally should just use mapping_dir
            path = os.path.join(mapping_dir, config['filename'])
            
            result_df = apply_and_save_mapping(result_df, [feat], path)
            if config['fillna'] is not None:
                result_df[feat].fillna(config['fillna'], inplace=True)
            
            category_feat.append(feat)
            
        # 2. Handle Angular Features (Sin/Cos)
        elif feat in ANGULAR_FEATURES:
            result_df, new_cols = process_angular_feature(result_df, feat)
            numerical_feat.extend(new_cols)

        # 3. Handle Numeric Features with NaN Indicators
        elif feat in NUMERIC_NA_CONFIG:
            fill_val = NUMERIC_NA_CONFIG[feat]
            result_df, new_cols = process_numeric_na_feature(result_df, feat, fill_val)
            numerical_feat.extend(new_cols)

        # 4. Handle Specific Custom Features
        elif feat == 'dssp_bend':
            result_df[feat] = result_df[feat].map({'S': 1})
            result_df[feat].fillna(0, inplace=True)
            numerical_feat.append(feat)
            logger.debug("dssp_bend: fillna(0) and map S to 1")

        elif feat == 'dssp_chirality':
            result_df[feat] = result_df[feat].map({'-': 1, '+': 2})
            result_df[feat].fillna(0, inplace=True)
            numerical_feat.append(feat)
            logger.debug("dssp_chirality: fillna(0) and map -/+ to 1/2")

        elif feat == 'ptms_mapped':
            result_df, new_cols = process_ptms(result_df)
            numerical_feat.extend(new_cols)

    return result_df, category_feat, numerical_feat


def process_angular_feature(df: pd.DataFrame, feat: str) -> Tuple[pd.DataFrame, List[str]]:
    """Handles Sin/Cos transformation and NaN indicator."""
    col_na = f'{feat.replace("dssp_", "")}_is_na'
    df[col_na] = df[feat].isna().astype('int8')
    
    rad = np.radians(df[feat])
    df[f'{feat}_sin'] = np.sin(rad).astype('float32')
    df[f'{feat}_cos'] = np.cos(rad).astype('float32')
    
    df[f'{feat}_sin'].fillna(0, inplace=True)
    df[f'{feat}_cos'].fillna(0, inplace=True)
    
    logger.debug(f"Processing {feat}: Sin/Cos conversion & {col_na} created")
    return df, [col_na, f'{feat}_sin', f'{feat}_cos']


def process_numeric_na_feature(df: pd.DataFrame, feat: str, fill_val: int) -> Tuple[pd.DataFrame, List[str]]:
    """Handles numeric features with NaN indicator and fill value."""
    suffix = feat.replace("dssp_", "").lower() + "_is_na"
    if 'TCO' in feat: 
        suffix = 'tco_is_na'

    df[suffix] = df[feat].isna().astype('int8')
    df[feat].fillna(fill_val, inplace=True)
    
    logger.debug(f"Processing {feat}: fillna({fill_val}) & {suffix}")
    return df, [suffix, feat] # Return both indicator and filled original column


def process_ptms(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Handles one-hot encoding of PTM lists."""
    sample_val = df['ptms_mapped'].dropna().iloc[0] if not df['ptms_mapped'].dropna().empty else None
    if isinstance(sample_val, str):
        df['ptms_mapped'] = df['ptms_mapped'].apply(parse_ptm)

    all_ptms = set()
    for l in df['ptms_mapped']:
        if isinstance(l, list):
            all_ptms.update(l)
    all_ptms_list = sorted(list(all_ptms))

    new_cols = []
    for ptm in all_ptms_list:
        col_name = f'ptm_{ptm}'
        # Safe check for list type + containment
        df[col_name] = df['ptms_mapped'].apply(lambda x: 1 if (isinstance(x, list) and ptm in x) else 0).astype('int8')
        new_cols.append(col_name)
        
    df['ptm_is_na'] = (df['ptms_mapped'].apply(len) == 0).astype('int8')
    new_cols.append('ptm_is_na')
    
    df.drop(columns=['ptms_mapped'], inplace=True)
    logger.debug(f"Processing ptms_mapped: One-hot encoding & ptm_is_na created")
    return df, new_cols


def make_bin_cols(
    df: pd.DataFrame, 
    gen_col_name: str, 
    bin_size: int = 42, 
    method: str = 'bin'
) -> pd.DataFrame:
    """
    Bins mutability data.
    method: 'bin' (equal width) or 'q' (quantile based)
    """
    if method == 'bin':
        df[gen_col_name] = pd.cut(df['mutability'], bins=bin_size, labels=False, include_lowest=True)
    elif method == 'q':
        df[gen_col_name] = pd.qcut(df['mutability'], q=bin_size, labels=False, duplicates='drop')

    df[gen_col_name] = df[gen_col_name].astype('int8')
    return df


def scaling_and_fillna_feature(df: pd.DataFrame, feat_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Scales and fills NaN values for specific features."""
    if feat_names is None:
        return df

    result_df = df.copy()

    for feat in feat_names:
        if "DAYM780301" in feat:
            result_df[feat] = 10**(result_df[feat]/10)
            result_df[feat].fillna(0, inplace=True)
        elif "HENS920102" in feat:
            result_df[feat] = 2**(result_df[feat]/3)
            result_df[feat].fillna(0, inplace=True)
        elif 'unique_patients_count' in feat:
            result_df[feat].fillna(0, inplace=True)
        elif 'total_mutations_count' in feat:
            result_df['mut_is_na'] = result_df[feat].isna().astype('int8')
            result_df[feat].fillna(0, inplace=True)
        elif 'unique_mutation_types_count' in feat:
            result_df[feat].fillna(0, inplace=True)
            
    return result_df


def apply_and_save_mapping(df: pd.DataFrame, columns: List[str], mapping_path: str) -> pd.DataFrame:
    """Applies categorical mapping, creating/updating mapping file if needed."""
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            full_mapping = json.load(f)
    else:
        full_mapping = {}

    for col in columns:
        df[col] = df[col].fillna('Unknown').astype(str)
        
        if col not in full_mapping:
            unique_labels = sorted(df[col].unique())
            if 'Unknown' in unique_labels:
                unique_labels.remove('Unknown')
            
            # Start indexing from 1 for valid labels, 0 for Unknown
            mapping = {label: i for i, label in enumerate(unique_labels, 1)}
            mapping['Unknown'] = 0
            
            full_mapping[col] = mapping

        # Map values
        unknown_idx = full_mapping[col].get('Unknown', 0)
        df[col] = df[col].map(lambda x: full_mapping[col].get(x, unknown_idx))

    if not os.path.exists(mapping_path):
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump(full_mapping, f, indent=4)
    
    return df


def parse_ptm(x: Any) -> List[Any]:
    """Parses PTM string representation to list."""
    if pd.isna(x) or x == '[]':
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse PTM: {x}")
            return []
    return x
