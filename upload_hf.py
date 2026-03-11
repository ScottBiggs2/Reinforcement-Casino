from huggingface_hub import HfApi
import os
import re

# --- 请修改以下参数 ---
repo_id = "YiyingXie/gemma-3-270m-it-dpo"  # 你的 Hugging Face 模型仓库路径
checkpoint_dir = "./checkpoints_google_gemma_3_270m_it_dpo"
# ---------------------

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# 1. 自动寻找最大的 Step 文件夹 (例如寻找名字里数字最大的 'checkpoint-250')
all_steps = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
# 提取数字并排序
step_numbers = [int(re.findall(r'\d+', s)[0]) for s in all_steps if re.findall(r'\d+', s)]

if step_numbers:
    latest_step = max(step_numbers)
    latest_folder_name = [s for s in all_steps if str(latest_step) in s][0]
    print(f"找到最新进度: {latest_folder_name}，将只上传此文件夹。")
    
    # 构建最终要上传的路径
    target_checkpoint_path = os.path.join(checkpoint_dir, latest_folder_name)
else:
    target_checkpoint_path = checkpoint_dir # 如果没找到子文件夹，就传整个目录

# 2. 执行上传 (增加 ignore_patterns 自动过滤冗余大文件)
folders_to_upload = [
    {"local": "./masks", "repo": "masks"},
    {"local": target_checkpoint_path, "repo": "final_checkpoint"}
]

for item in folders_to_upload:
    if os.path.exists(item["local"]):
        print(f"🚀 正在上传: {item['local']} -> {item['repo']}")
        api.upload_folder(
            folder_path=item["local"],
            path_in_repo=item["repo"],
            repo_id=repo_id,
            # --- 关键代码：忽略所有优化器状态和临时文件，只传模型权重 ---
            ignore_patterns=[
                "optimizer.pt", 
                "rng_state.pth", 
                "optimizer.bin",
                "scheduler.bin",
                "*.tmp"
            ]
        )
        print(f"✅ {item['repo']} 上传成功！")

print(f"\n✨ 上传结束！已自动为你节省数 GB 流量。")