import torch
import configs
from logger import Logger

# def compute_distance_matrix1(embeddings: torch.Tensor):
#     B, N, L = embeddings.shape
#     matrix = torch.zeros((B*N, B*N))

#     n_embs = B*N
#     embeddings = embeddings.view(B*N, L)
#     for i in range(n_embs):
#         j_0 = i + configs.PER_SUBJECT_SAMPLES - i%configs.PER_SUBJECT_SAMPLES
#         for j in range(j_0, n_embs):

#             matrix[i][j] = torch.pairwise_distance(embeddings[i], embeddings[j], p=2) 
    
#     # Logger.debug("Matrix", matrix)


def compute_distance_matrix(embeddings: torch.Tensor):
    
    dist_sq = 2  - 2 * embeddings @ embeddings.t()
    dist_sq = torch.clamp(dist_sq, min=0.0)  # numerical stability

    matrix = torch.sqrt(dist_sq)
    return matrix
