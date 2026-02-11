#!/bin/bash
#SBATCH --account=def-panch
#SBATCH --job-name=evol_pssm5
#SBATCH --output=logs/evol_pssm5_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=20:00:00

# load environment variables
source ~/.bashrc

# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023
module load openmpi/5.0.8

source $ENV_DIR/DRC/bin/activate

# Add tools to PATH
export PATH=$PATH:$PROJECT_DIR/ncbi-blast-2.17.0+/bin
export PATH=$PATH:$PROJECT_DIR/hhsuite-3.3.0-SSE2/bin

# Run the script
srun python data/ExtractAtt/evol_info.py \
    --raw_fasta ./data/reference/idmapping_2026_02_03.fasta.gz \
    --fasta_dir ./data/reference/fasta \
    --nr_db_path $PROJECT_DIR/nrdb \
    --uniref_db_path $PROJECT_DIR/data/UniRef30 \
    --pssm_dir $PROJECT_DIR/data/preprocess/pssm \
    --hmm_dir $PROJECT_DIR/data/preprocess/hmm \
    --workers 1 \
    --jobs pssm