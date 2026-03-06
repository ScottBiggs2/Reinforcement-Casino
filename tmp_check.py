import torch

def is_mlp(name):
    # MLP keywords same as cold_mask_finder
    keywords = ["gate_proj", "up_proj", "down_proj", "fc1", "fc2", "feed_forward", "ffn", "mlp.c_fc", "mlp.c_proj"]
    return any(k in name.lower() for k in keywords)

gt_path = "masks/warm_magnitude_google_gemma_3_270m_it_sparsity97.5pct.pt"
cold_path = "masks/cold_fisher_google_gemma_3_270m_it_qihoo360_Light-R1-DPOData_sparsity97.5pct_n256.pt"

gt = torch.load(gt_path, map_location="cpu")["masks"]
cold = torch.load(cold_path, map_location="cpu")["masks"]

mlp_params_total = 0
gt_mlp_kept = 0
gt_total_kept = 0
total_params = 0

for name, mask in gt.items():
    total_params += mask.numel()
    gt_total_kept += mask.sum().item()
    if is_mlp(name):
        mlp_params_total += mask.numel()
        gt_mlp_kept += mask.sum().item()

print(f"Total Params in GT: {total_params:,}")
print(f"Total Kept in GT: {gt_total_kept:,.0f} ({(gt_total_kept/total_params)*100:.2f}%)")
print(f"MLP Params in GT: {mlp_params_total:,}")
print(f"MLP Kept in GT: {gt_mlp_kept:,.0f} ({(gt_mlp_kept/mlp_params_total)*100:.2f}%)")

# Let's also check Jaccard manually just for MLPs
intersection = 0
union = 0
for name in cold.keys():
    if name in gt:
        c_m = cold[name].bool()
        g_m = gt[name].bool()
        intersection += (c_m & g_m).sum().item()
        union += (c_m | g_m).sum().item()

print(f"Intersection in MLPs: {intersection:,}")
print(f"Union in MLPs: {union:,}")
if union > 0:
    print(f"Jaccard in MLPs: {intersection/union:.4f}")
    
# Expected random intersection
expected_random_intersect = (gt_mlp_kept / mlp_params_total) * (cold[list(cold.keys())[0]].sum().item() / cold[list(cold.keys())[0]].numel() if len(cold)>0 else 0) * mlp_params_total
# Let's do it properly:
cold_kept_total = sum(m.sum().item() for m in cold.values())
cold_density = cold_kept_total / mlp_params_total
gt_mlp_density = gt_mlp_kept / mlp_params_total
exp_intersect = cold_density * gt_mlp_density * mlp_params_total
exp_union = cold_kept_total + gt_mlp_kept - exp_intersect
print(f"Expected Random Jaccard: {exp_intersect / exp_union:.4f}")

