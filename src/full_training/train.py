
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloader import get_dataloader
from utils.logging_utils import make_run_dir

def train(
    model_name="google/gemma-3-270m-it",
    n_steps=5,
    batch_size=2,
    learning_rate=5e-5,
    subset_size=50,
    run_name="gemma_full_training"
):
    """
    Train a Gemma model on the OpenR1 dataset for a few steps and log weight changes.
    """
    # Create a directory for this run
    run_dir = make_run_dir(base_dir="results", run_name=run_name)
    print(f"Run directory: {run_dir}")

    # Get the dataloader and tokenizer
    dataloader, tokenizer = get_dataloader(
        tokenizer_name=model_name,
        subset_size=subset_size,
        batch_size=batch_size
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Get the initial state of the model
    previous_state_dict = {name: p.clone().detach() for name, p in model.named_parameters()}

    # Training loop
    model.train()
    pbar = tqdm(total=n_steps, desc="Training")
    step = 0
    while step < n_steps:
        for batch in dataloader:
            if step >= n_steps:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(device)  # Use input_ids as labels for language modeling

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # --- Log weight changes ---
            current_state_dict = {name: p.clone().detach() for name, p in model.named_parameters()}
            weight_deltas = {name: current_state_dict[name] - previous_state_dict[name] for name in current_state_dict}

            # Save deltas for this step
            step_dir = os.path.join(run_dir, f"step_{step+1}")
            os.makedirs(step_dir, exist_ok=True)
            for name, delta in weight_deltas.items():
                # Skip saving deltas that are all zeros
                if torch.all(delta == 0):
                    continue
                safe_name = name.replace(".", "_")
                torch.save(delta, os.path.join(step_dir, f"{safe_name}.pt"))

            previous_state_dict = current_state_dict
            # -------------------------

            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
            step += 1

    pbar.close()
    print("Training finished.")

if __name__ == "__main__":
    # This allows running the script directly for testing
    # You might need to adjust the model name and other parameters
    train(model_name="google/-3-270m-it", n_steps=5)
