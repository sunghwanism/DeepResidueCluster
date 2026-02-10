#!/bin/bash
#SBATCH --account=def-panch
#SBATCH --job-name=permut_e5i3t13
#SBATCH --output=logs/permut_e5i3t13_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=130
#SBATCH --mem=30G
#SBATCH --time=05:00:00

# load environment variables
source ~/.bashrc

# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023
module load openmpi/5.0.8

source $ENV_DIR/DRC/bin/activate

export CLUSTERPATH=$PROJECT_DIR/results/DeepResidueNetwork/clustering

srun python analysis/runImportantPath.py \
                                        --savepath $PROJECT_DIR/results/DeepResidueNetwork/pathogenity/ \
                                        --node_path  $PROJECT_DIR/data/preprocess/exclubq/merge_all_features_v021026.csv \
                                        --cluster_path $CLUSTERPATH/ \
                                        --mutation_counts_col total_mutations_count \
                                        --MIN_MUTATIONS 0 \
                                        --N_PERMS 1000000 \
                                        --N_CORES 128 \
                                        --stratify mut+res \
                                        --bin 42 \
                                        --params e5i3t13



# python analysis/runImportantPath.py --params e5i3t15 \
#                                     --savepath $PROJECT_DIR/results/DeepResidueNetwork/pathogenity/ \
#                                     --node_path  $PROJECT_DIR/data/preprocess/exclubq/merge_all_features_v021026.csv \
#                                     --cluster_path $PROJECT_DIR/results/DeepResidueNetwork/clustering/ \
#                                     --mutation_counts_col total_mutations_count \
#                                     --N_PERMS 1000000 \
#                                     --N_CORES 128 \
#                                     --stratify mut+res \
#                                     --bin 42 \
#                                     --params e5i3t13