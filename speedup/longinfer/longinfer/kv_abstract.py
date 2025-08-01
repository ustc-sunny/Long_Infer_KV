import torch
import torch.nn.functional as F

def get_kv_abstract(kv_chunk):
    min_vector, _ = torch.min(kv_chunk, dim=0)
    max_vector, _ = torch.max(kv_chunk, dim=0)
    return min_vector, max_vector