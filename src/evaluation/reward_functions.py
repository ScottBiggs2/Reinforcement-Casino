from datasets import load_metric

def calculate_rouge(predictions, references):
    """
    Calculates ROUGE scores for a batch of predictions and references.

    Args:
        predictions (list of str): The generated responses from the model.
        references (list of str): The ground truth (chosen) responses.

    Returns:
        dict: A dictionary of ROUGE scores (e.g., rouge1, rouge2, rougeL).
    """
    rouge_metric = load_metric('rouge')
    scores = rouge_metric.compute(predictions=predictions, references=references)
    return scores

# You can add more reward functions here in the future.
# For example, a reward function based on a pre-trained reward model,
# or other metrics like BLEU, or even custom heuristics.

def placeholder_reward_function(predictions, references):
    """
    A placeholder for a future reward function.
    """
    # Example: reward based on length
    rewards = [len(p) for p in predictions]
    return {"length_reward": rewards}



    # To run:

    # python src/evaluation/DPO_evaluation.py \
    # --model_name_or_path google/gemma-3-270m-it \
    # --checkpoint_path /results/checkpoint/checkpoint-5 \
    # --mask_path /masks/top_10.0_percent_mask.pt \
    # --num_samples 20

    # python src/evaluation/DPO_evaluation.py --model_name_or_path google/gemma-3-270m-it --checkpoint_path /results/checkpoint/checkpoint-5 --mask_path /masks/top_10.0_percent_mask.pt --num_samples 20
