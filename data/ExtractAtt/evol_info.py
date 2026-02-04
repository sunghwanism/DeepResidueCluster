import os
import argparse
import gzip

import subprocess
from concurrent.futures import ProcessPoolExecutor

def split_fasta(input_path, output_dir):
    if os.path.exists(os.path.join(output_dir, "A0JLT2.fasta")):
        print("Split already done. Skipping.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    is_gzip = input_path.endswith('.gz')
    opener = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'

    try:
        with opener(input_path, mode, encoding='utf-8') as f:
            current_file = None
            for line in f:
                if line.startswith('>'):
                    parts = line.split('|')
                    id_part = parts[1] if len(parts) > 1 else line[1:].split()[0]
                    
                    if current_file:
                        current_file.close()
                    
                    save_path = os.path.join(output_dir, f"{id_part}.fasta")
                    current_file = open(save_path, 'w', encoding='utf-8')
                
                if current_file:
                    current_file.write(line)
            
            if current_file:
                current_file.close()

    except UnicodeDecodeError:
        print("Error: UnicodeDecodeError")


def extract_pssm(fasta_dir, pssm_dir, db_path,):
    os.makedirs(pssm_dir, exist_ok=True)

    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith(".fasta"):
            protein_id = fasta_file.split(".")[0]
            output_pssm = os.path.join(pssm_dir, f"{protein_id}.pssm")
            
            cmd = [
                "psiblast",
                "-query", os.path.join(fasta_dir, fasta_file),
                "-db", db_path,
                "-num_iterations", "3",
                "-evalue", "0.001",
                "-matrix", "BLOSUM62",
                "-out_ascii_pssm", output_pssm
            ]
            
            subprocess.run(cmd)

def run_single_psiblast(args):
    fasta_path, db_path, output_pssm = args
    
    cmd = [
        "psiblast",
        "-query", fasta_path,
        "-db", db_path,
        "-num_iterations", "3",
        "-evalue", "0.001",
        "-matrix", "BLOSUM62",
        "-out_ascii_pssm", output_pssm
    ]
    return subprocess.run(cmd, capture_output=True)

def extract_pssm_parallel(fasta_dir, pssm_dir, db_path, max_workers=192):
    os.makedirs(pssm_dir, exist_ok=True)
    tasks = []

    db_path = os.path.join(db_path, "nr") # NEED to add 'nr'

    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith(".fasta"):
            protein_id = fasta_file.split(".")[0]
            fasta_path = os.path.join(fasta_dir, fasta_file)
            output_pssm = os.path.join(pssm_dir, f"{protein_id}.pssm")
            tasks.append((fasta_path, db_path, output_pssm))

    print(f"Starting parallel PSSM extraction with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run_single_psiblast, tasks))

def run_single_hhblits(args):
    """
    args: (fasta_path, db_path, output_hhm)
    """
    fasta_path, db_path, output_hhm = args
    
    if os.path.exists(output_hhm):
        return f"Skipped: {os.path.basename(output_hhm)}"

    cmd = [
        "hhblits",
        "-i", fasta_path,
        "-d", db_path,
        "-ohhm", output_hhm,
        "-cpu", "1",
        "-n", "2",
        "-v", "0"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"Success: {os.path.basename(output_hhm)}"

    except subprocess.CalledProcessError as e:
        return f"Error in {os.path.basename(fasta_path)}: {e}"

def extract_hmm_parallel(fasta_files, db_path, hmm_dir, max_workers=192):
    os.makedirs(hmm_dir, exist_ok=True)
    
    tasks = []
    for fasta_path in fasta_files:
        protein_id = os.path.basename(fasta_path).replace(".fasta", "")
        output_hhm = os.path.join(hmm_dir, f"{protein_id}.hhm")
        tasks.append((fasta_path, db_path, output_hhm))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_hhblits, tasks))
    
    return results

# For Parallel Process
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_fasta", type=str, default=None, help="path to raw fasta file")
    parser.add_argument("--fasta_dir", type=str, required=True, help="path to fasta directory for saving")
    parser.add_argument("--nr_db_path", type=str, help="path to nr database")
    parser.add_argument("--uniref_db_path", type=str, help="path to uniref database")
    parser.add_argument("--pssm_dir", type=str, help="path to pssm directory for saving")
    parser.add_argument("--hmm_dir", type=str, help="path to hmm directory for saving")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="number of parallel workers")
    parser.add_argument('--jobs', nargs='+', choices=['split', 'pssm', 'hmm'], help="jobs to run [split, pssm, hmm]")
    args = parser.parse_args()

    if 'split' in args.jobs:
        assert args.raw_fasta is not None, "raw_fasta is required for split job"
        split_fasta(args.raw_fasta, args.fasta_dir)

    if 'pssm' in args.jobs:
        assert args.nr_db_path is not None, "nr_db_path is required for pssm job"
        assert args.pssm_dir is not None, "pssm_dir is required for pssm job"
        extract_pssm_parallel(args.fasta_dir, args.pssm_dir, args.nr_db_path, max_workers=args.workers)
        
    if 'hmm' in args.jobs:
        assert args.uniref_db_path is not None, "uniref_db_path is required for hmm job"
        assert args.hmm_dir is not None, "hmm_dir is required for hmm job"
        extract_hmm_parallel(args.fasta_dir, args.hmm_dir, args.uniref_db_path, max_workers=args.workers)