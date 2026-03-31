import torch

class Config:
    MODEL_PATH = "/home/asus/ycy/model/Meta-Llama-3-8B-Instruct"
    SAE_DIR = "/home/asus/ycy/sae_weights/Llama3_1-8B-Base-L15R-8x/checkpoints"
    SAE_PATH = f"{SAE_DIR}/final.safetensors"
    
    MAX_LENGTH = 512
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-5
    
    ENCODER_DIM = 4096
    EMBEDDING_DIM = 4096
    CIDT_HIDDEN_SIZE = 1024
    CIDT_LAYERS = 2
    
    # 临时特征索引，后续用因果分析替换
    _RAW_INDICES = [332, 20789, 11142, 24124, 6222, 21002, 9180, 21091, 7105, 2828, 28845, 30918, 20992, 21623, 868, 32427, 6831, 21296, 16267, 2886]
    VERIFIED_REFUSAL_INDICES = [idx for idx in _RAW_INDICES if idx < 32768]
    NUM_SAE_FEATURES = len(VERIFIED_REFUSAL_INDICES)
    
    # 关键：干预层和SAE训练层严格对齐
    INTERVENTION_LAYERS = [15]
    BASE_LAMBDA = 3.0
    DECAY_RATE = 0.01
    SVD_THRESHOLD = 1e-4
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16