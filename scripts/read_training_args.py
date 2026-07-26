import sys, torch
KEYS = ["max_completion_length","max_prompt_length","num_generations","beta",
        "learning_rate","max_steps","per_device_train_batch_size",
        "gradient_accumulation_steps","generation_batch_size"]
for p in sys.argv[1:]:
    try:
        a = torch.load(p + "/training_args.bin", map_location="cpu", weights_only=False)
    except Exception as e:
        print(p + "\n  ERROR " + type(e).__name__ + ": " + str(e) + "\n"); continue
    print(p.split("/transfer_v1/")[-1][:80])
    for k in KEYS:
        print("    {:<30} {}".format(k, getattr(a, k, "<absent>")))
    print()
