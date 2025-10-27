import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import torch

class MPSFixDPOTrainer(DPOTrainer):
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """
        Fix for MPS device where input_ids are incorrectly cast to float.
        """
        for key in batch:
            if "input_ids" in key and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].long()
        
        return super().get_batch_loss_metrics(model, batch, train_eval)
from datasets import load_dataset
import sys
import wandb
from transformers import TrainerCallback

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logging_utils import make_run_dir

# Custom callback to save weight deltas
class WeightDeltaCallback(TrainerCallback):
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.previous_state_dict = None

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.previous_state_dict is None:
            self.previous_state_dict = {name: p.clone().detach() for name, p in model.named_parameters()}

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        current_state_dict = {name: p.clone().detach() for name, p in model.named_parameters()}
        
        weight_deltas = {name: current_state_dict[name] - self.previous_state_dict[name] for name in current_state_dict}

        step_dir = os.path.join(self.run_dir, f"step_{int(state.global_step)}")
        os.makedirs(step_dir, exist_ok=True)
        for name, delta in weight_deltas.items():
            if torch.all(delta == 0):
                continue
            safe_name = name.replace(".", "_")
            torch.save(delta, os.path.join(step_dir, f"{safe_name}.pt"))
        
        self.previous_state_dict = current_state_dict

def my_extract_prompt(example):
    prompt = ""
    chosen = example["chosen"]
    rejected = example["rejected"]
    # Handle cases where chosen or rejected can be other than string
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return {"prompt": "", "chosen": str(chosen), "rejected": str(rejected)}
    for i in range(min(len(chosen), len(rejected))):
        if chosen[i] != rejected[i]:
            prompt = chosen[:i]
            break
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def train(
    model_name="google/gemma-3-270m-it",
    n_steps=5,
    batch_size=1, 
    learning_rate=5e-5,
    subset_size=10,
    run_name="gemma_dpo_training"
):
    """
    Train a Gemma model using DPOTrainer and log with wandb.
    """
    # Initialize wandb
    wandb.init(project="rl-casino", name=run_name, config={
        "model_name": model_name,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "subset_size": subset_size
    })

    # Create a directory for this run
    run_dir = make_run_dir(base_dir="results", run_name=run_name)
    print(f"Run directory: {run_dir}")

    # Load the dataset
    dataset = load_dataset("qihoo360/Light-R1-DPOData")
    if subset_size:
        dpo_dataset = dataset['train'].select(range(subset_size))
    else:
        dpo_dataset = dataset['train']
    
    dpo_dataset = dpo_dataset.map(my_extract_prompt)
    print("DPO dataset loaded and formatted.")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set up DPOConfig
    dpo_config = DPOConfig(
        output_dir=os.path.join(run_dir, "checkpoints"),
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=n_steps,
        logging_steps=1,
        report_to="wandb",
        remove_unused_columns=False,
        beta=0.1,
    )

    # Set up the DPOTrainer
    dpo_trainer = MPSFixDPOTrainer(
        model,
        ref_model=None, # DPOTrainer will create a reference model if None
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        callbacks=[WeightDeltaCallback(run_dir=run_dir)]
    )

    # Train the model
    print("Starting DPO training...")
    dpo_trainer.train()
    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    train()