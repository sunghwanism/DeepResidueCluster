#!/bin/bash
#SBATCH --account=def-panch
#SBATCH --job-name=MCL-e5i3t11
#SBATCH --output=logs/MCL-e5i3t11.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00


# load environment variables
source ~/.bashrc

# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023
module load openmpi/5.0.8

source $ENV_DIR/DRC/bin/activate


clustering_measures=(
    # DBSCAN
    MCL
    # HDBSCAN
    # DPClus
    # EAGLE
)

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

srun python src/models/Clustering/runCluster.py \
    --node_path  $PROJECT_DIR/data/preprocess/exclubq/node_mutation_with_BMR_v120525.csv \
    --graph_path mergedG_btw+clos+deg+pgr+spl.pkl \
    --savepath $PROJECT_DIR/results/DeepResidueNetwork/clustering/ \
    --measure "${clustering_measures[@]}"