import json, statistics as st
T="/scratch/xie.yiyi/transfer_v1"
runs={
 "dense_grpo(ckpt-500)": T+"/dense_grpo_math220k_llama8b/llama8b_math220k_dense_scott_full/checkpoints/checkpoint-500",
 "sparse_oracle_grpo":   T+"/sparse_grpo_math220k_oracle_grpo_math/sparse_grpo_math220k_oracle_grpo_math_500steps/checkpoints/checkpoint-500",
}
KEYS=["completions/clipped_ratio","completions/mean_terminated_length",
      "rewards/accuracy_reward/mean","rewards/accuracy_reward/std",
      "rewards/format_number_reward/std","rewards/format_reasoning_reward/std",
      "reward","reward_std","frac_reward_zero_std","grad_norm","kl"]
for name,p in runs.items():
    h=[e for e in json.load(open(p+"/trainer_state.json")).get("log_history",[]) if "reward" in e]
    print("="*78); print(f"{name}   logged steps: {len(h)}")
    if not h: continue
    print("{:<40}{:>10}{:>10}{:>10}".format("metric","mean","median","max"))
    for k in KEYS:
        v=[e[k] for e in h if k in e and e[k] is not None]
        if v: print("{:<40}{:>10.4f}{:>10.4f}{:>10.4f}".format(k, st.mean(v), st.median(v), max(v)))
    z=[e.get("frac_reward_zero_std",0) for e in h if "frac_reward_zero_std" in e]
    if z:
        dead=sum(1 for x in z if x>=0.999)
        print(f"\n  steps with frac_reward_zero_std == 1.0 (ZERO gradient): {dead}/{len(z)} = {100*dead/len(z):.1f}%")
        print(f"  mean fraction of each batch with no signal: {100*st.mean(z):.1f}%")
    acc=[e.get("rewards/accuracy_reward/std",0) for e in h if "rewards/accuracy_reward/std" in e]
    rs =[e.get("reward_std",0) for e in h if "reward_std" in e]
    if acc and rs:
        share=[a/r for a,r in zip(acc,rs) if r>1e-9]
        if share: print(f"  accuracy_std / reward_std (share of the LEARNING signal): mean {st.mean(share):.3f}")
    print()
