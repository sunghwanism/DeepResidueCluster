#!/bin/bash
#SBATCH --account=ctb-panch_gpu
#SBATCH --job-name=DGI-emb_Training-
#SBATCH --output=logs/DGI-emb_Training-.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00

# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023 nvhpc/25.1 openmpi/5.0.3
module load cuda/12.6

srun nproc
srun nvidia-smi

source $SCRATCH/shmoon/envs/DRC/bin/activate
srun which pip

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# NEED to set for wandb
export WANDB_RUN_ID=""
export WANDB_API_KEY=""
export ENTITY_NAME=""

### Save your wandb API key in your .bash_profile or replace $API_KEY with your actual API key.
### Uncomment the line below and comment out "wandb offline" if running in online mode ###

# wandb login $WANDB_API_KEY 
srun python script/train.py \
     --config_path config/run.yaml \
     --wandb_key $WANDB_API_KEY \
     --entity_name $ENTITY_NAME \
     --project_name DeepResidueCluster \
     --wandb_run_name $WANDB_RUN_ID \
     --batch_size 512 \
     --num_workers 8 \
     --use_aug \
     # --load_pretrained \
     # --wandb_run_id $WANDB_RUN_ID \