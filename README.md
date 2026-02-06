# DeepResidueCluster

# Dependency
- torch==2.7.1
- CUDA==12.6
- pytorch-geometric==2.7.0
- networkx==3.5

# Environment
Create a virtual environment and install the dependencies
Start from envs directory
```bash
module load python/3.11
virtualenv --no-download ../envs/DRC
source ../envs/DRC/bin/activate
pip install --no-index --upgrade pip
```

# Preprocessing
## Evolutionary Information
At first, you need to download FASTA format file of your target protein through [UniProt](https://www.uniprot.org/id-mapping).
Input your protein ID in the blank box and download the FASTA format file.

### PSSM (Position Specific Scoring Matrix)
PSSM takes two days with 192 CPU and 300GB RAM (Parallel CPU Processing)
```bash
# DownLoad BLAST
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.17.0+-x64-linux.tar.gz
tar -zxvf ncbi-blast-2.17.0+-x64-linux.tar.gz

# Add BLAST to PATH
export PATH=$PATH:/{YOURPATH}/ncbi-blast-2.17.0+/bin

# Unzip Reference Database
mkdir -p ./db/nr # Make dir for NR database
cd ./db/nr
../../bin/update_blastdb.pl --decompress nr # Take Long time (Need over 100GB)
```

Then, you run the code below. (This code support Parallel CPU Processing)
```bash
python data/ExtractAtt/evol_info.py \
--raw_fasta ENTER_YOUR_FASTA_FILE_PATH # FASTAFILE downloaded from UniProt
--fasta_dir ./data/reference/fasta
--nr_db_path ENTER_REFERENCE_DATABASE_PATH
--workers NUM_CPUS
--pssm_dir SAVE_RESULT_PATH
--jobs split pssm # if you don't have to run split job, you only input pssm
```

### HHM (Hidden Markov model)
HMM takes 4 hours with 40 CPU and 400GB RAM (Parallel CPU Processing)
```bash
# Download HMM 
mkdir hhsuite-3.3.0-SSE2
cd hhsuite-3.3.0-SSE2
wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz

# Download UniRef30 Database
mkdir -p ./db/uniref30_db
cd ./db/uniref30_db
wget http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/UniRef30_2023_02_hhsuite.tar.gz
tar -xvzf UniRef30_2023_02_hhsuite.tar.gz
export PATH=$PATH:/{YOURPATH}/hhsuite-3.3.0-SSE2/bin
```

Then, you run the code below. (This code support Parallel CPU Processing)

```bash
python data/ExtractAtt/evol_info.py \
--raw_fasta ENTER_YOUR_FASTA_FILE_PATH # FASTAFILE downloaded from UniProt
--fasta_dir ./data/reference/fasta
--uniref_db_path ENTER_REFERENCE_DATABASE_PATH
--workers NUM_CPUS
--hmm_dir SAVE_RESULT_PATH
--jobs split hmm # if you don't have to run split job, you only input pssm
```


# Data Structure
The features should be annotated in the graph as node attributes. Used features are specified in the config file.
Example: 
- Graph: NetworkX graph object
    - Node features: ```degree```, ```pagerank```, ```closeness```, ```avgShortestPath```, ```betweenness```

- Features: DataFrame object
    - Mutation Related Features (5)
    - Residue Location Features (7)
    - Residue Property Features (3)
    - Constant Features (1)

# Execution
```bash
sbatch script/sbatch.sh
```