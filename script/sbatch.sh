#!/bin/bash
#SBATCH --job-name=DGI-sanity
#SBATCH --output=logs/DGI-sanity_%j.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=23:59:59
#SBATCH --partition=compute
#SBATCH --mail-type=FAIL

# move to base PATH
cd $SCRATCH/shmoon/DeepResidueCluster

# Load module
module load StdEnv/2023  nvhpc/25.1  openmpi/5.0.3
module load cuda/12.6

srun nvidia-smi

source $SCRATCH/shmoon/envs/DRC/bin/activate

# For CUDA-Calucation
export NX_CUGRAPH_AUTOCONFIG=True
export NETWORKX_FALLBACK_TO_NX=True # Change GPU to CPU, if GPU is not available

# NEED to set for wandb
export WANDB_RUN_ID=your_wandb_run_id_here
export WANDB_API_KEY=your_wandb_api_key_here


srun python script/train.py \
     --config_path config/run.yaml \
     --wandb_key $WANDB_API_KEY \
     --entity_name Panchenko-Lab \
     --project_name DeepResidueCluster \
     --wandb_run_name DGI \
     --wandb_run_id $WANDB_RUN_ID \
     --batch_size 32 \
     --num_workers 8 \
     # --load_pretrained  \
     # --nowandb