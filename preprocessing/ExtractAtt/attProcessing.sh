#!/bin/bash
#SBATCH --account=def-panch
#SBATCH --job-name=get_node_att(inter)
#SBATCH --output=logs/get_node_att(inter).txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=100G
#SBATCH --time=06:00:00

# Load bashrc
source ~/.bashrc

cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6
module load suitesparse/7.6.0
module load gcc/12.3

# Run virtual env
source $ENV_DIR/DRC/bin/activate

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MEASURES=(
    # "degree"
    # "transitivity"
    # "triangles"
    # "k_truss"
    # "k_core"
    # "pagerank"
    "closeness"
    "betweenness"
    "shortest_path_length_per_node"
    # "eigenvector"
    # "local_clustering"
    # "bridges"
    # "biconnected_components"
)

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# Set your paths
export SAVEPATH= #ADD SAVE PATH
export GRAPHPATH= #ADD GRAPHPATH (pkl format)

srun python preprocessing/ExtractAtt/run.py \
    --graph_path $GRAPHPATH \
    --param_path centralityConfig.py \
    --savepath $SAVEPATH \
    --measure "${MEASURES[@]}"