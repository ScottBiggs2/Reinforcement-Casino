import torch, json
T="/scratch/xie.yiyi/transfer_v1"
runs={
 "dense_grpo": T+"/dense_grpo_math220k_llama8b/llama8b_math220k_dense_scott_full/checkpoints/checkpoint-500",
 "sparse_oracle_grpo": T+"/sparse_grpo_math220k_oracle_grpo_math/sparse_grpo_math220k_oracle_grpo_math_500steps/checkpoints/checkpoint-500",
 "sparse_orc_dpo_lr1": T+"/sparse_grpo_math220k_oracle_dpo_lightr1_500/sparse_grpo_math220k_oracle_dpo_lightr1_500_500steps/checkpoints/checkpoint-500",
}
K=["max_steps","lr_scheduler_type","warmup_ratio","warmup_steps","learning_rate","num_train_epochs"]
vals={n: torch.load(p+"/training_args.bin", map_location="cpu", weights_only=False) for n,p in runs.items()}
print("{:<22}".format("field") + "".join("{:>24}".format(n) for n in runs))
for k in K:
    print("{:<22}".format(k) + "".join("{:>24}".format(str(getattr(a,k,"?"))) for a in vals.values()))
print()
for n,p in runs.items():
    ts=json.load(open(p+"/trainer_state.json"))
    lrs=[e["learning_rate"] for e in ts.get("log_history",[]) if "learning_rate" in e]
    gs=ts.get("global_step")
    print("{:<22} global_step={:<6} lr_at_start={:<12} lr_at_step500={}".format(
        n, str(gs), str(lrs[0] if lrs else None), str(lrs[-1] if lrs else None)))
