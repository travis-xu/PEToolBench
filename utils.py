import json
import re
import torch
import transformers
import transformers.models.llama.modeling_llama
from functools import partial
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def load_json(file_name, file_path=""):
    file = []
    with open(file_path + file_name, "r") as f:
        file = json.load(f)
    return file

def save_json(data, file_name, file_path=""):
    with open(file_path + file_name, "w") as f:
        json.dump(data, f)