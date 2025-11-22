import os
import json
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
