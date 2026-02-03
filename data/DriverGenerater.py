import os
import re
import ast

import numpy as np
import pandas as pd

from collections import Counter
from reference import residue1to3


def merge_and_average_dicts(dict_list):
    total_sum = Counter()   
    total_count = Counter() 
    
    for d in dict_list:
        for key, value in d.items():
            total_sum[key] += int(value)
            total_count[key] += 1
            
    average_dict = {key: total_sum[key] / total_count[key] for key in total_sum}
    return average_dict


def _load_cosmic_status_map(dataPath: str, score_th: float) -> dict:
    def parse_pos(v):
        match = re.search(r"c.(\d+)([A-Z])", v)
        if match:
            return int(match.group(1))
        return None

    def parse_residue(v):
        match = re.search(r"c.(\d+)([A-Z])", v)
        if match:
            return residue1to3[match.group(2).upper()].lower()
        return None


    if score_th > 1:
        score_th /= 100

    driver = pd.read_csv(dataPath, sep='\t', compression='gzip')

    driver = driver[['Accession Number','FATHMM score', 'Mutation CDS',]]
    driver.dropna(subset=['FATHMM score'], inplace=True, axis=0)
    driver['label'] = driver['FATHMM score'] > score_th
    driver['res'] = driver['Mutation CDS'].apply(parse_residue)
    driver['pos'] = driver['Mutation CDS'].apply(parse_pos)
    driver.drop(['Mutation CDS'], inplace=True, axis=1)
    driver = driver[driver['Accession Number'].str.startswith("ENST")]

    agg_df = driver.groupby(['Accession Number', 'pos', 'res']).agg(['sum', 'count'])['label']
    agg_df['driver_idx'] = agg_df['sum']/agg_df['count']
    agg_df['is_driver'] = agg_df['driver_idx'].apply(lambda x: 1 if x >0 else 0)
    agg_df = agg_df[['is_driver']].reset_index()

    status_map = {}
    for idx, row in agg_df.iterrows():
        status_map[(row['Accession Number'], int(row['pos']), row['res'])] = row['is_driver']

    return status_map
    

def _load_mutagene_status_map(dataPath: str) -> dict:
    def parse_pos(v):
        match = re.search(r"([A-Z])(\d+)", v)
        if match:
            return int(match.group(2))
        return None

    def parse_residue(v):
        match = re.search(r"([A-Z])(\d+)", v)
        if match:
            return residue1to3[match.group(1).upper()].lower()
        return None

    def extract_ensmbl_id(val):
        if not isinstance(val, str):
            return None
        for part in val.split(';'):
            clean_part = part.strip()
            if clean_part.startswith('ENST'):
                return clean_part

    mutagene = pd.read_csv(dataPath)
    mutagene['label'] = mutagene['label'].map({'Passenger':0, 'Driver':1})
    mutagene.dropna(subset=['label', 'Transcript ID'], inplace=True, axis=0)
    mutagene['ENST'] = mutagene['Transcript ID'].apply(extract_ensmbl_id)
    mutagene.dropna(subset=['ENST'], inplace=True, axis=0)
    mutagene.drop(['Transcript ID', 'Protein ID', 'PubMed Id', 'Swissprot ID'], axis=1, inplace=True)

    mutagene['pos'] = mutagene['mutation'].apply(parse_pos)
    mutagene['res'] = mutagene['mutation'].apply(parse_residue)

    agg_df = mutagene.groupby(['ENST', 'pos', 'res']).agg(['sum', 'count'])['label']
    agg_df['driver_idx'] = agg_df['sum']/agg_df['count']
    agg_df['is_driver'] = agg_df['driver_idx'].apply(lambda x: 1 if x >0 else 0)
    agg_df = agg_df[['is_driver']].reset_index()

    status_map = {}
    for idx, row in agg_df.iterrows():
        status_map[(row['ENST'], int(row['pos']), row['res'])] = row['is_driver']

    return status_map


def _load_chemplus_status_map(path: str) -> dict:

    def parse_res(v):
        match = re.search(r"p\.([A-Z])(\d+)", v)
        if match:
            return residue1to3[match.group(1).upper()].lower()

    def parse_pos(v):
        match = re.search(r"p\.([A-Z])(\d+)", v)
        if match:
            return int(match.group(2))

    df = pd.read_excel(path, skiprows=2)
    df = df[['Transcript ID', 'gwCHASMplus score', 'mutation', 'gwCHASMplus score', 'Novel']]
    df['res'] = df['mutation'].apply(parse_res)
    df['pos'] = df['mutation'].apply(parse_pos)

    status_map = {}
    for idx, row in df.iterrows():
        status_map[(row['Transcript ID'], int(row['pos']), row['res'])] = 1
        
    return status_map

def getDriver_df(
    dataPath: dict[str, str], 
    all_node_df: pd.DataFrame, 
    score_th: float, 
    reference_data: str
) -> pd.DataFrame:
    """
    Annotates the node dataframe with driver information.
    
    reference_data: 'COSMIC', 'MutaGene', or 'all'.
    
    is_driver Classification:
    0: Passenger
    1: Driver
    2: Both / Conflict
    """
    
    status_map_mutagene = {}
    status_map_cosmic = {}
    status_map_chemplus = {}
    use_mutagene = False
    use_cosmic = False
    use_chemplus = False

    status_map_list = []

    if reference_data == 'all':
        paths = dataPath

        path_mg = paths['MutaGene']
        path_co = paths['COSMIC']
        path_cp = paths['ChemPlus']
        
        status_map_mutagene = _load_mutagene_status_map(path_mg)
        status_map_cosmic = _load_cosmic_status_map(path_co, score_th)
        status_map_chemplus = _load_chemplus_status_map(path_cp)
        
        use_mutagene = True
        use_cosmic = True
        use_chemplus = True
        status_map_list = [status_map_mutagene, status_map_cosmic, status_map_chemplus]
        
    elif reference_data == 'MutaGene':
        status_map_mutagene = _load_mutagene_status_map(dataPath['MutaGene'])
        use_mutagene = True
        status_map_list.append(status_map_mutagene)
        
    elif reference_data == 'COSMIC':
        status_map_cosmic = _load_cosmic_status_map(dataPath['COSMIC'], score_th)
        use_cosmic = True
        status_map_list.append(status_map_cosmic)

    elif reference_data == 'ChemPlus':
        status_map_chemplus = _load_chemplus_status_map(dataPath['ChemPlus'])
        use_chemplus = True
        status_map_list.append(status_map_chemplus)

    merged_status_map = merge_and_average_dicts(status_map_list)
    driver_map_df = map_driver_status(all_node_df, merged_status_map)
    driver_map_df.dropna(subset=['is_driver'], inplace=True, axis=0)
        
    return driver_map_df

def parse_ids(x):
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
        except:
            raise ValueError(f"Invalid value in ensmbl_id: {x}")
    else:
        val = x

    for i, ensmbl_id in enumerate(val):
        val[i] = ensmbl_id.split('.')[0]

    return val

def map_driver_status(df, score_dict):
    result_df = df.copy()
    result_df['ensmbl'] = result_df['ensmbl_id'].apply(parse_ids) 

    def calculate_row_status(row):
        ensmbl_list = row['ensmbl']
        try:
            position = int(row['position'])
        except (ValueError, TypeError):
            return np.nan
            
        residue = row['residueType']
        
        valid_values = []

        for ensmbl_id in ensmbl_list:
            key = (ensmbl_id, position, residue)
            val = score_dict.get(key)

            if val is not None:
                mapped_val = 1 if val > 0 else 0
                valid_values.append(mapped_val)
        
        if not valid_values:
            return np.nan

        final_score = sum(valid_values) / len(ensmbl_list)
        return 1 if final_score > 0 else 0

    result_df['is_driver'] = result_df.apply(calculate_row_status, axis=1)
    
    return result_df