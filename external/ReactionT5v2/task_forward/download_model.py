import os
# 1. 设置国内镜像环境变量（必须放在导入 huggingface_hub 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 定义模型列表（你需要用到哪个就下载哪个）
models = [
    "sagawa/ReactionT5v2-forward",          # 正向反应预测
    # "sagawa/ReactionT5v2-retrosynthesis", # 逆合成预测 (如果需要请取消注释)
    # "sagawa/ReactionT5v2-yield"           # 产率预测 (如果需要请取消注释)
]

print("开始下载模型...")

for model_id in models:
    # 将模型名中的斜杠替换为下划线，作为本地文件夹名
    local_dir = f"./weights/{model_id.split('/')[-1]}"
    
    print(f"正在下载: {model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 确保下载的是真实文件而不是快捷方式
        resume_download=True          # 支持断点续传
    )

print("下载完成！")