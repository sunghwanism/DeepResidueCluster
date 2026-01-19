import yaml
import os
import sys


from utils.functions import init_wandb

# Add the project root to the python path to allow imports from models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    # Load run configuration
    run_config_path = os.path.join(os.path.dirname(__file__), '../config/run.yaml')
    with open(run_config_path, 'r') as f:
        run_config = yaml.safe_load(f)
    
    model_name = run_config.get('model')
    DATABASE = run_config.get('DATA_PATH')
    
    init_wandb(run_config)
    
    if model_name == 'DGI':
        from models.DGI.execute import run_training
        dgi_config_path = os.path.join(os.path.dirname(__file__), '../config/DGI.yaml')
        run_training(dgi_config_path)
    else:
        print(f"Unknown model: {model_name}")

if __name__ == '__main__':
    main()
