#!/bin/bash
#SBATCH --job-name=DGI-emb_Training-1
#SBATCH --output=logs/DGI-emb_Training-1.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=compute


# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023 nvhpc/25.1 openmpi/5.0.3
module load cuda/12.6

srun nproc
srun nvidia-smi

source $SCRATCH/shmoon/envs/DRC/bin/activate

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# NEED to set for wandb
export WANDB_RUN_ID="Training-1"
export WANDB_API_KEY=
export ENTITY_NAME=

### Save your wandb API key in your .bash_profile or replace $API_KEY with your actual API key.
### Uncomment the line below and comment out "wandb offline" if running in online mode ###

# wandb login $WANDB_API_KEY 
srun python script/train.py \
     --config_path config/run.yaml \
     --wandb_key $WANDB_API_KEY \
     --entity_name $ENTITY_NAME \
     --project_name DeepResidueCluster \
     --wandb_run_name $WANDB_RUN_ID \
     --wandb_run_id $WANDB_RUN_ID \
     --batch_size 32 \
     --num_workers 16 \
     --nowandb \
     --use_aug \
     # --load_pretrained  \