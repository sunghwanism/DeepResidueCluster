#!/bin/bash
#SBATCH --job-name=DataGenerator
#SBATCH --output=logs/DataGenerator.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --partition=debug

# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023  nvhpc/25.1  openmpi/5.0.3

source $SCRATCH/shmoon/envs/DRC/bin/activate

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# NEED to set for wandb
export WANDB_RUN_NAME="EnterYourRunName"
export WANDB_API_KEY="EnterYourWandbApiKey"
export ENTITY_NAME="EnterYourWandbEntityName"

srun python script/runData.py \
     --config_path config/run.yaml \
     --wandb_key $WANDB_API_KEY \
     --entity_name $ENTITY_NAME \
     --project_name DeepResidueCluster \
     --wandb_run_name $WANDB_RUN_NAME \
     --batch_size 256 \
     --num_workers 16 \
     --nowandb \
     --use_aug