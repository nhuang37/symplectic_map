import subprocess
import itertools
import os

def run_grid_search(script_path, base_args=None, 
                    num_layers_list=[8, 16, 32], 
                    hid_dim_list=[4, 8, 16], 
                    lr_list=[1e-3, 1e-4], 
                    log_dir="results_clean_search/grid_logs"):
    """
    Run a grid search over specified hyperparameters.

    Parameters:
    - script_path (str): Path to the training script.
    - base_args (list): Additional arguments as a list of strings.
    - num_layers_list, hid_dim_list, lr_list (list): Hyperparameter values to search.
    - log_dir (str): Directory to store logs.
    """

    os.makedirs(log_dir, exist_ok=True)
    base_args = base_args or []

    combinations = list(itertools.product(num_layers_list, hid_dim_list, lr_list))
    for i, (num_layers, hid_dim, lr) in enumerate(combinations):
        run_name = f"layers{num_layers}_hid{hid_dim}_lr{lr}"
        log_file = os.path.join(log_dir, f"{run_name}.log")

        cmd = ["python", script_path] + base_args + [
            "--num_layers", str(num_layers),
            "--hid_dim", str(hid_dim),
            "--lr", str(lr)
        ]
        with open(log_file, "w") as outfile:
            print(f"Running: {' '.join(cmd)} -> {log_file}")
            subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT)

# Example usage:
run_grid_search("train_clean.py")#, base_args=["--num_epochs", "20"])
