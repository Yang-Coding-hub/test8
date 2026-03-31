from safetensors.torch import load_file
import os

# 你的SAE根目录
SAE_DIR = "/home/asus/ycy/sae_weights/Llama3_1-8B-Base-L15R-8x/checkpoints"

# 1. 扫描目录结构
print("===== 目录下的所有文件 =====")
for f in os.listdir(SAE_DIR):
    f_path = os.path.join(SAE_DIR, f)
    size_mb = os.path.getsize(f_path) / (1024 * 1024)
    print(f"{f:40s} | Size: {size_mb:.2f} MB")

# 2. 打印safetensors权重详情
safetensor_files = [f for f in os.listdir(SAE_DIR) if f.endswith(".safetensors")]
for file_name in safetensor_files:
    file_path = os.path.join(SAE_DIR, file_name)
    weights = load_file(file_path)
    print(f"\n===== 权重文件：{file_name} 详情 =====")
    for key, tensor in weights.items():
        print(f"键名: {key:30s} | Shape: {tuple(tensor.shape)} | Dtype: {tensor.dtype}")