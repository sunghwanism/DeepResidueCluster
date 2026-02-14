import re
from functools import lru_cache

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Iterable

import ast
import time

import pandas as pd
import requests

from reference.residue_dictionary import residue3to1
from tqdm.auto import tqdm

ENSEMBL_REST = "https://rest.ensembl.org"

NODE_PATTERN = re.compile(r"^([A-Za-z0-9\-]+)_(-?\d+)_([A-Za-z]{3})$")

CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def prefetch_uniprot_to_ensp(node_ids: Iterable[str]) -> Dict[str, List[str]]:
    unique_uniprots = set()
    for node_id in node_ids:
        uniprot_id, _, _ = parse_node_id(node_id)
        if uniprot_id:
            unique_uniprots.add(uniprot_id)

    return {u: uniprot_to_ensp(u) for u in sorted(unique_uniprots)}


def _safe_get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    headers = {"Content-Type": "application/json"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if not r.ok:
        return None
    try:
        return r.json()
    except Exception:
        return None


def strip_copy_index(uniprot_like_id: str) -> str:
    # p06899-4 -> p06899 (homodimer copy index 제거)
    return re.sub(r"-\d+$", "", uniprot_like_id.lower())


def parse_node_id(node_id: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    m = NODE_PATTERN.match(node_id.strip())
    if not m:
        return None, None, None
    raw_uniprot, pos_str, aa3 = m.groups()
    uniprot_id = strip_copy_index(raw_uniprot)
    pos = int(pos_str)
    aa1 = residue3to1.get(aa3.upper())

    return uniprot_id.upper(), pos, aa1


@lru_cache(maxsize=20000)
def uniprot_to_ensp(uniprot_id: str) -> List[str]:
    # Use UniProt API to map UniProt ID -> Ensembl Protein ID (ENSP)
    # The Ensembl REST API /xrefs/id/ endpoint often fails for UniProt IDs.
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?fields=xref_ensembl"
    
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
        if not r.ok:
            return []
        data = r.json()
    except Exception:
        return []

    xrefs = data.get("uniProtKBCrossReferences", [])
    ensp_ids = set()
    
    for xref in xrefs:
        if xref.get("database") == "Ensembl":
            for prop in xref.get("properties", []):
                if prop.get("key") == "ProteinId":
                    val = prop.get("value")
                    if val and val != "-":
                        ensp_id = val.split('.')[0]
                        ensp_ids.add(ensp_id)
    
    return sorted(list(ensp_ids))


@lru_cache(maxsize=50000)
def ensp_to_enst(ensp_id: str) -> Optional[str]:
    # ENSP -> Parent transcript (ENST)
    url = f"{ENSEMBL_REST}/lookup/id/{ensp_id}"
    data = _safe_get_json(url, params={"expand": 0})
    if not data:
        return None
    parent = data.get("Parent")
    return parent if isinstance(parent, str) and parent.startswith("ENST") else None


@lru_cache(maxsize=50000)
def get_protein_seq(ensp_id: str) -> Optional[str]:
    url = f"{ENSEMBL_REST}/sequence/id/{ensp_id}"
    data = _safe_get_json(url, params={"type": "protein"})
    if not data:
        return None
    seq = data.get("seq")
    return seq.upper() if isinstance(seq, str) else None


@lru_cache(maxsize=50000)
def get_cds_seq(enst_id: str) -> Optional[str]:
    # ENST00000374948.6 -> ENST00000374948
    enst_no_ver = enst_id.split(".")[0]
    url = f"{ENSEMBL_REST}/sequence/id/{enst_no_ver}"
    data = _safe_get_json(url, params={"type": "cds"})
    if not data:
        return None
    seq = data.get("seq")
    time.sleep(0.2)
    return seq.upper() if isinstance(seq, str) else None


def cds_context_5nt(cds_seq: str, pos_1based: int) -> Optional[str]:
    if pos_1based <= 0:
        return None

    codon_start_idx = (pos_1based - 1) * 3
    
    if codon_start_idx >= len(cds_seq):
        return None

    start = codon_start_idx - 1
    end = codon_start_idx + 4
    
    if start < 0 or end > len(cds_seq):
        return None

    context = cds_seq[start:end]
    
    return context if len(context) == 5 else None

def find_matching_ensts(
    uniprot_id: str,
    aa_pos: int,
    aa1: str,
    uniprot_ensp_map: Optional[Dict[str, List[str]]] = None
) -> List[str]:
    if not uniprot_id or not aa1 or aa_pos <= 0:
        return []

    ensp_candidates = (
        uniprot_ensp_map.get(uniprot_id, [])
        if uniprot_ensp_map is not None
        else uniprot_to_ensp(uniprot_id)
    )

    matched_enst = []
    for ensp in ensp_candidates:
        pseq = get_protein_seq(ensp)
        time.sleep(0.1)
        
        if not pseq or aa_pos > len(pseq):
            continue

        if pseq[aa_pos - 1] != aa1:
            continue
        enst = ensp_to_enst(ensp)
        if enst:
            matched_enst.append(enst)

    return sorted(set(matched_enst))

def build_node_context_df(node_ids: List[str]) -> pd.DataFrame:

    rows = []
    uniprot_ensp_map = prefetch_uniprot_to_ensp(node_ids)

    for node_id in tqdm(node_ids, desc="Processing nodes"):
        uniprot_id, aa_pos, aa1 = parse_node_id(node_id)

        if not uniprot_id or aa_pos is None or not aa1:
            rows.append({
                "node_id": node_id,
                "ensmbl_id": None,
                "cds_contexts": None,
                "unique_cds_contexts": None
            })
            continue

        enst_list = find_matching_ensts(uniprot_id, aa_pos, aa1, uniprot_ensp_map)

        if not enst_list:
            rows.append({
                "node_id": node_id,
                "ensmbl_id": None,
                "cds_contexts": None,
                "unique_cds_contexts": None
            })
            continue

        cds_contexts = []
        for enst in enst_list:
            cds = get_cds_seq(enst)
            cds_contexts.append(cds_context_5nt(cds, aa_pos) if cds else None)

        uniq = sorted({c for c in cds_contexts if c is not None}) or None

        rows.append({
            "node_id": node_id,
            "ensmbl_id": enst_list,
            "cds_contexts": cds_contexts,
            "unique_cds_contexts": uniq
        })

    return pd.DataFrame(rows)

def translate_dna(codon):
    return CODON_TABLE.get(codon, 'X')

def get_nonsynonymous_mutability(five_mer, mutation_freq_dict):
    
    if len(five_mer) != 5:
        return 0.0

    b1, b2, b3, b4, b5 = list(five_mer)
    original_codon = b2 + b3 + b4
    original_aa = translate_dna(original_codon)
    
    mutability_score = 0.0

    context1 = b1 + b2 + b3
    if context1 in mutation_freq_dict:
        for mutated_seq, freq in mutation_freq_dict[context1].items():
            mutated_b2 = mutated_seq[1] # 가운데 글자가 변이된 염기
            new_codon = mutated_b2 + b3 + b4
            if translate_dna(new_codon) != original_aa:
                mutability_score += freq

    context2 = b2 + b3 + b4
    if context2 in mutation_freq_dict:
        for mutated_seq, freq in mutation_freq_dict[context2].items():
            mutated_b3 = mutated_seq[1]
            new_codon = b2 + mutated_b3 + b4
            if translate_dna(new_codon) != original_aa:
                mutability_score += freq

    context3 = b3 + b4 + b5
    if context3 in mutation_freq_dict:
        for mutated_seq, freq in mutation_freq_dict[context3].items():
            mutated_b4 = mutated_seq[1]
            new_codon = b2 + b3 + mutated_b4
            if translate_dna(new_codon) != original_aa:
                mutability_score += freq
                
    return mutability_score


def calculate_mutability_for_row(row, mutation_freq):
    if pd.isna(row.get('unique_cds_contexts')):
        return None

    try:
        val_unique = row['unique_cds_contexts']
        if isinstance(val_unique, str):
            unique_cds_context = list(ast.literal_eval(val_unique))
        else:
            unique_cds_context = list(val_unique)
            
        val_full = row['cds_contexts']
        if isinstance(val_full, str):
            cds_contexts = list(ast.literal_eval(val_full))
        else:
            cds_contexts = list(val_full)
            
    except (ValueError, SyntaxError):
        return None

    cds_counts = Counter(cds_contexts)
    total_count = len(cds_contexts)
    
    if total_count == 0:
        return 0.0

    total_mutability_sum = 0.0
    
    for cds in unique_cds_context:
        context_mutability = get_nonsynonymous_mutability(cds, mutation_freq)
        weight = cds_counts[cds] / total_count
        total_mutability_sum += (context_mutability * weight)

    return total_mutability_sum