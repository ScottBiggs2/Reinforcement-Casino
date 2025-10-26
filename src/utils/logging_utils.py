# src/utils/logging_utils.py
import os
import json
import datetime

def make_run_dir(base_dir="results", run_name=None):
    """Create a timestamped results directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"run_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_config(config, run_dir, filename="config.json"):
    """Save experiment configuration as JSON."""
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path

def save_results(results, run_dir, filename="results.json"):
    """Save results dictionary as JSON."""
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path

def save_mask(mask_dict, run_dir, layer_name="layer0", filename=None):
    """Save a neuron/weight mask as JSON."""
    filename = filename or f"mask_{layer_name}.json"
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        json.dump(mask_dict, f, indent=2)
    return path
