# DeepResidueCluster

# Dependency
- torch==2.7.1
- CUDA==12.6
- pytorch-geometric==2.7.0
- networkx==3.5

# Environment
At first, load the modules for the environment
```bash
module load StdEnv/2023  nvhpc/25.1  openmpi/5.0.3
module load cuda/12.6
```
Then, create a virtual environment and install the dependencies
```bash
module load python/3.11
virtualenv --no-download DRC
source DRC/bin/activate
pip install --no-index --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1
pip install torch-geometric==2.7.0
```

# Execution
```bash
module load StdEnv/2023  nvhpc/25.1  openmpi/5.0.3
module load cuda/12.6
```