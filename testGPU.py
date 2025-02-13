import torch

# 获取可用的设备数
device_count = torch.cuda.device_count()

# 确保设备 ID 小于设备数
device_id = 0  # 假设你选择 GPU 0
if device_id < device_count:
    device = torch.device(f"cuda:{device_id}")
else:
    device = torch.device("cpu")  # 如果设备 ID 不有效，则选择 CPU

print(f"Using device: {device}")
