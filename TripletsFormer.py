import torch
import configs
from logger import Logger


def compute_distance_matrix(embeddings: torch.Tensor):
    
    cosine_sim = embeddings @ embeddings.t()
    return 1 - cosine_sim





def get_a_n_p_pairs(distance_matrix):
    triplets = []  
    for i in range(len(distance_matrix)):
        row = distance_matrix[i].clone()
        
        # Find the range of images for the same person
        person_start = (i // configs.PER_SUBJECT_SAMPLES) * configs.PER_SUBJECT_SAMPLES
        person_end = person_start + configs.PER_SUBJECT_SAMPLES
        
        # Get all possible positives for this anchor (same person, excluding self)
        same_person_indices = list(range(person_start, person_end))
        same_person_indices.remove(i)  # Remove anchor itself
        
        # Mask out same person's embeddings for negative selection
        row[person_start:person_end] = float('inf')
        
        # Get top-k hardest negatives (you can adjust k)
        k_hardest_negatives = 3  # Adjust this number based on your needs
        _, neg_indices = torch.topk(row, k_hardest_negatives, largest=False)
        
        # Create triplets: anchor with each positive and each hard negative
        for pos_idx in same_person_indices:
            for neg_idx in neg_indices:
                triplets.append((i, pos_idx, neg_idx.item()))  # (anchor, positive, negative)
    
    return triplets

def form_triplets(embeddings, triplets):
    anchors = []
    positives = []
    negatives = []
    
    for anchor_idx, pos_idx, neg_idx in triplets:
        anchors.append(embeddings[anchor_idx])
        positives.append(embeddings[pos_idx])
        negatives.append(embeddings[neg_idx])
        
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return anchors, positives, negatives
