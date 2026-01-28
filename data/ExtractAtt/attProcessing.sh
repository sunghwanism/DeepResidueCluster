#!/bin/bash
#SBATCH --job-name=get_node_att
#SBATCH --ntasks=1
#SBATCH --output=logs/get_node_att.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=23:59:59
#SBATCH --partition=compute
#SBATCH --mail-type=FAIL

# move to base PATH
cd /home/moonsun6/links/scratch/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6
module load suitesparse/7.6.0
module load gcc/12.3

# Run virtual env
source $SCRATCH/shmoon/envs/DRC/bin/activate

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

EASY=(
    # "degree"
    # "transitivity"
    # "triangles"
    # "k_truss"
    # "k_core"
    # "pagerank"
    "closeness"
    # "betweenness"
    # "shortest_path_length_per_node"
    # "eigenvector"
    # "local_clustering"
    # "bridges"
    # "biconnected_components"
)

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# Set your paths
export SAVEPATH= #Add your save path here (directory)
export NODEPATH= #Add your node path here (csv File)
export GRAPHPATH= #Add your graph path here (pkl File)

srun python data/ExtractAtt/run.py \
    --node_path $NODEPATH \
    --graph_path $GRAPHPATH \
    --param_path centralityConfig.py \
    --savepath $SAVEPATH \
    --measure "${EASY[@]}"