import json, statistics as st
p="/scratch/xie.yiyi/transfer_v1/dense_grpo_math220k_llama8b/llama8b_math220k_dense_scott_full/checkpoints/checkpoint-500/trainer_state.json"
h=[e for e in json.load(open(p))["log_history"] if "reward" in e]
def blk(rows,label):
    g=lambda k:[r[k] for r in rows if k in r and r[k] is not None]
    cr,tl=g("completions/clipped_ratio"),g("completions/mean_terminated_length")
    am,asd=g("rewards/accuracy_reward/mean"),g("rewards/accuracy_reward/std")
    rsd,fz=g("reward_std"),g("frac_reward_zero_std")
    share=[a/r for a,r in zip(asd,rsd) if r>1e-9]
    print(f"{label:<26}{st.mean(cr):>10.3f}{st.mean(tl):>12.1f}{st.mean(am):>12.4f}"
          f"{(st.mean(share) if share else float('nan')):>14.3f}{st.mean(fz):>12.3f}")
print("{:<26}{:>10}{:>12}{:>12}{:>14}{:>12}".format(
    "window (cap=1024)","clipped","term_len","acc_mean","acc_signal%","zero_std"))
blk(h[:20],  "first 20 steps")
blk(h[:100], "first 100 steps")
blk(h,       "all 500 steps")
