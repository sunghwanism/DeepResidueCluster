import ast
import pandas as pd
import numpy as np
import os

from reference import residue1to3


def _load_cosmic_status_map(dataPath: str, score_th: float) -> dict:
    """
    Loads COSMIC data and returns a map of {(ID, position, residue_name): status}.
    Status:
    0: Passenger (Score <= th)
    1: Driver (Score > th)
    2: Both (Site contains both Passenger and Driver mutations)
    """
    print(f"Loading COSMIC data from {dataPath}...")
    try:
        DriverInfo = pd.read_csv(dataPath, sep='\t', compression='gzip')
    except Exception as e:
        print(f"Error loading COSMIC: {e}. Trying without compression...")
        DriverInfo = pd.read_csv(dataPath, sep='\t')
        
    # User modification: Adjust score threshold if provided as percentage-like?
    # Original user code: if score_th > 0: score_th /= 100
    if score_th > 0:
        score_th /= 100
        
    # Select columns and filter empty CDS
    df = DriverInfo[['Gene name', 'Accession Number', 'FATHMM score', 'Mutation CDS', 'GRCh']].dropna(subset=['Mutation CDS']).copy()

    # Extract Position
    df['position'] = df['Mutation CDS'].str.extract(r'(\d+)', expand=False)
    df.dropna(subset=['position'], inplace=True)
    df['position'] = df['position'].astype(int)
    
    # Extract Residue (Amino Acid)
    df['residue_name_raw'] = df['Mutation CDS'].str.extract(r'\d+([A-Z])>', expand=False)
    df['residue_name'] = df['residue_name_raw'].map(residue1to3).fillna(np.nan)
    
    if df['residue_name'].isnull().any():
            df['residue_name'] = df['residue_name'].fillna('G')
    
    df['residue_name'] = df['residue_name'].str.lower()
    df['is_driver_mutation'] = df['FATHMM score'] > score_th
    
    grouped = df.groupby(['Accession Number', 'position', 'residue_name'])['is_driver_mutation'].agg(['any', 'all'])
    
    status_map = {}
    
    for idx, row in grouped.iterrows():
        has_driver = row['any']
        all_drivers = row['all']
        has_passenger = not all_drivers # If not all are drivers, then some are passengers (since bool).
        
        if has_driver and has_passenger:
            status = 2
        elif has_driver:
            status = 1
        else: # No driver, only passengers (since 'any' is False)
            status = 0
            
        status_map[idx] = status
        
    return status_map


def _load_mutagene_status_map(dataPath: str) -> dict:
    """
    Loads MutaGene data and returns map of {(ID, position, residue_name): status}.
    MutaGene entries assumed to be Drivers (1).
    """
    print(f"Loading MutaGene data from {dataPath}...")
    mutagene_df = pd.read_csv(dataPath, sep=',')
    mutagene_key = 'Transcript ID'
    
    if mutagene_key not in mutagene_df.columns:
            print(f"Warning: {mutagene_key} not in MutaGene columns. Columns: {mutagene_df.columns}")
            return {}

    mutagene_df = mutagene_df.dropna(subset=[mutagene_key], axis=0).copy()

    def extract_ensmbl_id(val):
        if not isinstance(val, str):
            return []
        ids = []
        for part in val.split(';'):
            clean_part = part.strip()
            if clean_part.startswith('ENST'):
                ids.append(clean_part)
        return ids if ids else None

    mutagene_df['ENST'] = mutagene_df[mutagene_key].apply(extract_ensmbl_id)
    mutagene_df.dropna(subset=['ENST'], inplace=True)
    mutagene_df = mutagene_df.explode('ENST')
    
    mutagene_df['position'] = mutagene_df['mutation'].apply(lambda x: int(x[1:-1]))
    mutagene_df['residue_name_raw'] = mutagene_df['mutation'].apply(lambda x: x[0])
    mutagene_df['residue_name'] = mutagene_df['residue_name_raw'].map(lambda x: residue1to3.get(x.upper(), np.nan)).fillna(np.nan)
    
    mutagene_df['residue_name'] = mutagene_df['residue_name'].str.lower()
    
    mutagene_df.dropna(subset=['residue_name'], inplace=True)
    
    # Parse Label Column
    # Check if label column exists
    if 'label' not in mutagene_df.columns:
        print("Warning: 'label' column not found in MutaGene data. Assuming all are Drivers (1).")
        mutagene_df['status'] = 1
    else:
        # Map label to status
        def parse_label(val):
            if not isinstance(val, str): return 1 # Default or skip? Assuming 1 per original assumption?
            val = val.lower()
            if 'Driver' in val: return 1
            if 'Passenger' in val: return 0
            return 1 # Fallback? Or None?
        
        mutagene_df['status'] = mutagene_df['label'].apply(parse_label)

    # Group by Site and Aggregate
    grouped = mutagene_df.groupby(['ENST', 'position', 'residue_name'])['status'].agg(['min', 'max'])
    # If min=0 and max=1 -> Both (2)
    # If min=1 and max=1 -> Driver (1)
    # If min=0 and max=0 -> Passenger (0)
    
    status_map = {}
    for idx, row in grouped.iterrows():
        mn = row['min']
        mx = row['max']
        if mn == 0 and mx == 1:
            status_map[idx] = 2
        elif mx == 1:
            status_map[idx] = 1
        elif mx == 0:
            status_map[idx] = 0
            
    return status_map


def getDriver_df(
    dataPath: str, 
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
    use_mutagene = False
    use_cosmic = False

    if reference_data == 'all':
        paths = dataPath.split(',')
        if len(paths) != 2:
            print("Error: For reference_data='all', dataPath must be comma-separated strings for MutaGene and COSMIC files respectively.")
            return all_node_df
        
        path_mg = paths[0].strip()
        path_co = paths[1].strip()
        
        status_map_mutagene = _load_mutagene_status_map(path_mg)
        status_map_cosmic = _load_cosmic_status_map(path_co, score_th)
        use_mutagene = True
        use_cosmic = True
        
    elif reference_data == 'MutaGene':
        status_map_mutagene = _load_mutagene_status_map(dataPath)
        use_mutagene = True
        
    elif reference_data == 'COSMIC':
        status_map_cosmic = _load_cosmic_status_map(dataPath, score_th)
        use_cosmic = True

    # Annotate
    drop_na_all_node_df = all_node_df.dropna(subset=['unique_cds_contexts']).copy()
    print(f"Nodes with valid CDS context: {len(drop_na_all_node_df)}")

    def parse_ids(x):
        if isinstance(x, list): return x
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list): return val
                return [x]
            except:
                return [x]
        return []

    def check_driver_status(row):
        pos = row['position']
        res = row['residue_name']
        raw_ids = row['ensmbl_id']
        ids_list = parse_ids(raw_ids)
        
        # We need to collect valid statuses from all matching IDs
        # A node might map to multiple ENSTs. 
        # If ANY ENST says 2, status is 2.
        # If ONE ENST says 1 and another says 0 -> Conflict -> 2.
        
        collected_statuses = set()
        
        for eid in ids_list:
            clean_eid = str(eid).split('.')[0]
            key = (clean_eid, pos, res)
            
            # Get status from sources
            s_mg = status_map_mutagene.get(key, None)
            s_co = status_map_cosmic.get(key, None)
            
            # Resolve ID-level status
            current_status = None
            
            if use_mutagene and use_cosmic:
                # Merge Logic
                # If both exist
                if s_mg is not None and s_co is not None:
                    if s_mg == s_co:
                        current_status = s_mg
                    else:
                        # Conflict (e.g. 1 and 0, or 2 and 1) -> 2
                        current_status = 2
                elif s_mg is not None:
                    current_status = s_mg
                elif s_co is not None:
                    current_status = s_co
                else:
                    current_status = 0 # Neither found -> assume not driver (so 0? Or None?)
                    # If not found in DB, is it 0 (Passenger) or just unclassified?
                    # The request implies 0 is Passenger.
                    # Usually "Not in Driver DB" means effectively Passenger or Unknown.
                    # Assuming 0.
            
            elif use_mutagene:
                current_status = s_mg if s_mg is not None else 0
            elif use_cosmic:
                current_status = s_co if s_co is not None else 0
            
            if current_status is not None:
                 collected_statuses.add(current_status)

        # Aggregate collected statuses across IDs
        if not collected_statuses:
            return 0
            
        if 2 in collected_statuses:
            return 2
        if 1 in collected_statuses and 0 in collected_statuses:
            return 2 # Mixed -> 2
        if 1 in collected_statuses:
            return 1
        return 0

    if (use_mutagene and not status_map_mutagene) and (use_cosmic and not status_map_cosmic):
        print("Warning: No driver keys found.")
        drop_na_all_node_df['is_driver'] = 0
    else:
        drop_na_all_node_df['is_driver'] = drop_na_all_node_df.apply(check_driver_status, axis=1)

    # Stats
    if 'is_driver' in drop_na_all_node_df.columns:
        counts = drop_na_all_node_df['is_driver'].value_counts()
        print("Driver Counts:")
        print(counts)
        
    return drop_na_all_node_df