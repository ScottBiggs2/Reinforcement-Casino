from transformers import AutoModelForCausalLM, AutoTokenizer

def load_gemma(model_name="google/gemma-3-270m-it", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype="auto"  # FP16 if on MPS/CPU
    )
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_gemma()
    print("Model and tokenizer loaded successfully.")