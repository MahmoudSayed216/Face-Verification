import torch
import configs
from logger import Logger


def compute_distance_matrix(embeddings: torch.Tensor):
    
    dist_sq = 2  - 2 * embeddings @ embeddings.t()
    dist_sq = torch.clamp(dist_sq, min=0.0)  # numerical stability

    matrix = torch.sqrt(dist_sq)
    return matrix
