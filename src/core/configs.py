import torch
import os
from src.core.paths import VERIFIED_REFUSAL_FEATURES_PT


def _load_dynamic_refusal_indices(default_indices):
    if not VERIFIED_REFUSAL_FEATURES_PT.exists():
        return default_indices
    try:
        obj = torch.load(VERIFIED_REFUSAL_FEATURES_PT, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "top_indices" in obj:
            vals = [int(x) for x in obj["top_indices"]]
            return vals if vals else default_indices
    except Exception:
        return default_indices
    return default_indices

class Config:
    # Allow overriding from environment for portability across machines.
    MODEL_PATH = os.getenv("MODEL_PATH", "/home/asus/ycy/model/Meta-Llama-3-8B-Instruct")
    SAE_DIR = os.getenv("SAE_DIR", "/home/asus/ycy/sae_weights/Llama3_1-8B-Base-L15R-8x/checkpoints")
    SAE_PATH = f"{SAE_DIR}/final.safetensors"
    
    MAX_LENGTH = 512
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-5
    ROUTER_ENTROPY_LAMBDA = 1e-3
    ROUTER_ALIGN_LAMBDA = 0.1
    ROUTER_TEMPERATURE = 1.0
    RISK_CALIBRATION_BIAS = 0.9
    RISK_CALIBRATION_TEMP = 0.35
    REWRITE_RISK_THRESHOLD = float(os.getenv("REWRITE_RISK_THRESHOLD", "0.58"))
    CIDT_VERBOSE_LOG = os.getenv("CIDT_VERBOSE_LOG", "1") == "1"
    CIDT_LOG_EVERY = int(os.getenv("CIDT_LOG_EVERY", "50"))
    
    ENCODER_DIM = 4096
    EMBEDDING_DIM = 4096
    CIDT_HIDDEN_SIZE = 1024
    CIDT_LAYERS = 1
    
    # 临时特征索引，后续用因果分析替换
    _RAW_INDICES = [3179, 11627, 28533, 9478, 22531, 25500, 6132, 27905, 455, 17987, 9594, 3152, 31726, 10161, 17778, 6687, 868, 32053, 21435, 23347, 407, 4241, 14236, 5737, 9639, 32132, 23808, 1010, 28746, 3342, 5823, 13104, 8959, 13065, 22100, 20490, 29080, 9430, 9455, 8666, 7431, 7944, 4881, 10555, 14715, 2606, 16709, 14918, 6222, 20402, 4634, 347, 13175, 16344, 17067, 25227, 6589, 8930, 21600, 3285, 13102, 15508, 4863, 26740]
    _RAW_INDICES = _load_dynamic_refusal_indices(_RAW_INDICES)
    VERIFIED_REFUSAL_INDICES = _RAW_INDICES
    VERIFIED_REFUSAL_INDICES = [idx for idx in _RAW_INDICES if idx < 32768]
    NUM_SAE_FEATURES = len(VERIFIED_REFUSAL_INDICES)
    
    # 关键：干预层和SAE训练层严格对齐
    INTERVENTION_LAYERS = [15]
    BASE_LAMBDA = 1.2
    MAX_INTERVENTION_SCALE = 1.5
    DECAY_RATE = 0.01
    SVD_THRESHOLD = 1e-4
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16