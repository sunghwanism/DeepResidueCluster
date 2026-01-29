# DeepResidueCluster

# Dependency
- torch==2.7.1
- CUDA==12.6
- pytorch-geometric==2.7.0
- networkx==3.5

# Environment
Create a virtual environment and install the dependencies
Start from envs directory
```bash
module load python/3.11
virtualenv --no-download ../envs/DRC
source ../envs/DRC/bin/activate
pip install --no-index --upgrade pip
pip install -e .
```

# Data Structure
The features should be annotated in the graph as node attributes. Used features are specified in the config file.
Example: 
- Graph: NetworkX graph object
    - Node features: ```degree```, ```pagerank```, ```closeness```, ```avgShortestPath```, ```betweenness```

- Features: DataFrame object
    - Mutation Related Features (5)
    - Residue Location Features (7)
    - Residue Property Features (3)
    - Constant Features (1)

# Execution
```bash
sbatch script/sbatch.sh
```