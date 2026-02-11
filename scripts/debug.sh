#!/bin/bash
#SBATCH --job-name=DGI-emb_sanity_check3
#SBATCH --output=logs/DGI-emb_sanity_check3.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=debug

# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023  nvhpc/25.1  openmpi/5.0.3
module load cuda/12.6

srun nproc
srun nvidia-smi

source $SCRATCH/shmoon/envs/DRC/bin/activate

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# NEED to set for wandb
export WANDB_RUN_NAME="EnterYourRunName"
export WANDB_API_KEY="EnterYourWandbApiKey"
export ENTITY_NAME="EnterYourWandbEntityName"

# [Important]
# If you want to load_pretrained model, you need to set WANDB_RUN_ID
# export WANDB_RUN_ID=""

srun python script/train.py \
     --config_path config/run.yaml \
     --wandb_key $WANDB_API_KEY \
     --entity_name $ENTITY_NAME \
     --project_name DeepResidueCluster \
     --wandb_run_name $WANDB_RUN_NAME \
     --batch_size 256 \
     --num_workers 16 \
     --nowandb \
     --use_aug
     # --load_pretrained  \
     # --wandb_run_id $WANDB_RUN_ID \